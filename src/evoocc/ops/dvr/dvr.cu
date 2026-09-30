#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

// upper bound on voxels traversed per ray (700 + 700 + 45 + 1)
#define MAX_D 1446
#define MAX_STEP 1000

enum LossType {L1, L2, ABSREL};
enum PhaseName {TEST, TRAIN};

template <typename scalar_t>
__global__ void init_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> occupancy) {

    const auto n = blockIdx.y;

    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    const auto M = points.size(1);
    const auto T = occupancy.size(1);

    if (c < M) {
        const auto t = tindex[n][c];

        assert(T == 1 || t < T);

        if (t < 0) return;

        const auto ts = (T == 1) ? 0 : t;

        const int vzsize = occupancy.size(2);
        const int vysize = occupancy.size(3);
        const int vxsize = occupancy.size(4);

        const int vx = int(points[n][c][0]);
        const int vy = int(points[n][c][1]);
        const int vz = int(points[n][c][2]);

        if (0 <= vx && vx < vxsize &&
            0 <= vy && vy < vysize &&
            0 <= vz && vz < vzsize) {
            occupancy[n][ts][vz][vy][vx] = 1;
        }
    }
}

template <typename scalar_t>
__global__ void render_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> coord_index,
    PhaseName train_phase) {

    const auto n = blockIdx.y;

    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    const auto M = points.size(1);
    const auto T = sigma.size(1);

    if (c < M) {
        const auto t = tindex[n][c];

        assert(T == 1 || t < T);

        const auto ts = (T == 1) ? 0 : t;

        if (t < 0) return;

        const int vzsize = sigma.size(2);
        const int vysize = sigma.size(3);
        const int vxsize = sigma.size(4);

        const double xo = origin[n][t][0];
        const double yo = origin[n][t][1];
        const double zo = origin[n][t][2];

        const double xe = points[n][c][0];
        const double ye = points[n][c][1];
        const double ze = points[n][c][2];

        const int vxo = int(xo);
        const int vyo = int(yo);
        const int vzo = int(zo);

        const int vxe = int(xe);
        const int vye = int(ye);
        const int vze = int(ze);

        int vx = vxo;
        int vy = vyo;
        int vz = vzo;

        const double rx = xe - xo;
        const double ry = ye - yo;
        const double rz = ze - zo;
        double gt_d = sqrt(rx * rx + ry * ry + rz * rz);

        const double dx = rx / gt_d;
        const double dy = ry / gt_d;
        const double dz = rz / gt_d;

        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;
        const int stepZ = (dz >= 0) ? 1 : -1;

        const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
        const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX;
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX;
        double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX;

        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
        const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

        int3 path[MAX_D];
        // cumulative sigma*delta
        double csd[MAX_D];
        // per-voxel alpha (hit probability)
        double p[MAX_D];
        double d[MAX_D];

        int step = 0;
        int count = 0;
        double last_d = 0.0;

        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) &&
                (0 <= vy && vy < vysize) &&
                (0 <= vz && vz < vzsize);
            if (inside) {
                was_inside = true;
                path[count] = make_int3(vx, vy, vz);
            } else if (was_inside) {
                break;
            }
            double _d = 0.0;
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    _d = tMaxX;
                    vx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    _d = tMaxY;
                    vy += stepY;
                    tMaxY += tDeltaY;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
            if (inside) {
                const int3 &v = path[count];
                const double _sigma = sigma[n][ts][v.z][v.y][v.x];
                // clamp is important for correctness
                const double _delta = max(0.0, _d - last_d);
                const double sd = _sigma * _delta;
                if (count == 0) {
                    csd[count] = sd;
                    p[count] = 1 - exp(-sd);
                } else {
                    csd[count] = csd[count-1] + sd;
                    p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                }
                d[count] = _d;
                count ++;
            }
            last_d = _d;
            step ++;

            if (step > MAX_STEP) {
                break;
            }
        }

        assert(count <= MAX_D);
        
        if (count > 0) {
            double exp_d = d[count-1];
            
            const int3 &v_init = path[count-1];
            int x = v_init.x;
            int y = v_init.y;
            int z = v_init.z;

            for (int i = 0; i < count; i++) {
                const int3 &v = path[i];
                const double occ = sigma[n][ts][v.z][v.y][v.x];
                if (occ > 0.5) {
                    exp_d = d[i];
                    
                    x = v.x;
                    y = v.y;
                    z = v.z;
                
                    break;
                }

            }

            double p_out = exp(-csd[count-1]);
            double max_d = d[count-1];


            if (train_phase == 1) {
                gt_d = min(gt_d, max_d);
            }

            pred_dist[n][c] = exp_d;
            gt_dist[n][c] = gt_d;
          
            coord_index[n][c][0] = double(x);
            coord_index[n][c][1] = double(y);
            coord_index[n][c][2] = double(z);

        }
    }
}

// sigma: N x T x H x L x W, origin: N x T x 3, points: N x M x 4; returns dist: N x M
std::vector<torch::Tensor> render_forward_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid,
    std::string phase_name) {

    const auto N = points.size(0);
    const auto M = points.size(1);

    const auto T = grid[0];
    const auto H = grid[1];
    const auto L = grid[2];
    const auto W = grid[3];

    const auto device = sigma.device();

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);


    auto gt_dist = -torch::ones({N, M}, device);
    auto pred_dist = -torch::ones({N, M}, device);

    auto coord_index = torch::zeros({N, M, 3}, device);

    PhaseName train_phase;
    if (phase_name.compare("test") == 0) {
        train_phase = TEST;
    } else if (phase_name.compare("train") == 0){
        train_phase = TRAIN;
    } else {
        std::cout << "UNKNOWN PHASE NAME: " << phase_name << std::endl;
        exit(1);
    }

    AT_DISPATCH_FLOATING_TYPES(sigma.scalar_type(), "render_forward_cuda", ([&] {
                render_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    origin.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    gt_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    coord_index.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    train_phase);
            }));

    cudaDeviceSynchronize();

    return {pred_dist, gt_dist, coord_index};
}

template <typename scalar_t>
__global__ void render_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_sigma,
    LossType loss_type) {

    const auto n = blockIdx.y;

    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    const auto M = points.size(1);
    const auto T = sigma.size(1);

    if (c < M) {
        const auto t = tindex[n][c];

        assert(T == 1 || t < T);

        const auto ts = (T == 1) ? 0 : t;

        if (t < 0) return;

        const int vzsize = sigma.size(2);
        const int vysize = sigma.size(3);
        const int vxsize = sigma.size(4);

        const double xo = origin[n][t][0];
        const double yo = origin[n][t][1];
        const double zo = origin[n][t][2];

        const double xe = points[n][c][0];
        const double ye = points[n][c][1];
        const double ze = points[n][c][2];

        const int vxo = int(xo);
        const int vyo = int(yo);
        const int vzo = int(zo);

        const int vxe = int(xe);
        const int vye = int(ye);
        const int vze = int(ze);

        int vx = vxo;
        int vy = vyo;
        int vz = vzo;

        const double rx = xe - xo;
        const double ry = ye - yo;
        const double rz = ze - zo;
        double gt_d = sqrt(rx * rx + ry * ry + rz * rz);

        const double dx = rx / gt_d;
        const double dy = ry / gt_d;
        const double dz = rz / gt_d;

        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;
        const int stepZ = (dz >= 0) ? 1 : -1;

        const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
        const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX;
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX;
        double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX;

        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
        const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

        int3 path[MAX_D];
        double csd[MAX_D];
        double p[MAX_D];
        double d[MAX_D];
        double dt[MAX_D];

        int step = 0;
        int count = 0;
        double last_d = 0.0;

        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) &&
                (0 <= vy && vy < vysize) &&
                (0 <= vz && vz < vzsize);
            if (inside) {
                was_inside = true;
                path[count] = make_int3(vx, vy, vz);
            } else if (was_inside) {
                break;
            } else if (last_d > gt_d) {
                break;
            }
            double _d = 0.0;
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    _d = tMaxX;
                    vx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    _d = tMaxY;
                    vy += stepY;
                    tMaxY += tDeltaY;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
            if (inside) {
                const int3 &v = path[count];
                const double _sigma = sigma[n][ts][v.z][v.y][v.x];
                const double _delta = max(0.0, _d - last_d);
                const double sd = _sigma * _delta;
                if (count == 0) {
                    csd[count] = sd;
                    p[count] = 1 - exp(-sd);
                } else {
                    csd[count] = csd[count-1] + sd;
                    p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                }
                d[count] = _d;
                dt[count] = _delta;
                count ++;
            }
            last_d = _d;
            step ++;

            if (step > MAX_STEP) {
                break;
            }
        }

        assert(count <= MAX_D);

        if (count > 0) {
            double exp_d = 0.0;
            for (int i = 0; i < count; i ++)
                exp_d += p[i] * d[i];

            double p_out = exp(-csd[count-1]);
            double max_d = d[count-1];

            exp_d += (p_out * max_d);
            gt_d = min(gt_d, max_d);

            pred_dist[n][c] = exp_d;
            gt_dist[n][c] = gt_d;

            double dd_dsigma[MAX_D];
            for (int i = count - 1; i >= 0; i --) {
                if (i == count - 1)
                    dd_dsigma[i] = p_out * max_d;
                else
                    dd_dsigma[i] = dd_dsigma[i+1] - exp(-csd[i]) * (d[i+1] - d[i]);
            }

            for (int i = count - 1; i >= 0; i --)
                dd_dsigma[i] *= dt[i];

            for (int i = count - 1; i >= 0; i --)
                dd_dsigma[i] -= dt[i] * p_out * max_d;

            double dl_dd = 1.0;
            if (loss_type == L1)
                dl_dd = (exp_d >= gt_d) ? 1 : -1;
            else if (loss_type == L2)
                dl_dd = (exp_d - gt_d);
            else if (loss_type == ABSREL)
                dl_dd = (exp_d >= gt_d) ? (1.0/gt_d) : -(1.0/gt_d);

            for (int i = 0; i < count; i ++) {
                const int3 &v = path[i];
                grad_sigma[n][ts][v.z][v.y][v.x] += dl_dd * dd_dsigma[i];
            }
        }
    }
}

// sigma: N x T x H x L x W, origin: N x T x 3, points: N x M x 4; returns dist/loss: N x M, grad_sigma: N x T x H x L x W
std::vector<torch::Tensor> render_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex,
    std::string loss_name) {

    const auto N = points.size(0);
    const auto M = points.size(1);

    const auto device = sigma.device();

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    auto gt_dist = -torch::ones({N, M}, device);
    auto pred_dist = -torch::ones({N, M}, device);
    auto grad_sigma = torch::zeros_like(sigma);

    LossType loss_type;
    if (loss_name.compare("l1") == 0) {
        loss_type = L1;
    } else if (loss_name.compare("l2") == 0) {
        loss_type = L2;
    } else if (loss_name.compare("absrel") == 0) {
        loss_type = ABSREL;
    } else if (loss_name.compare("bce") == 0){
        loss_type = L1;
    } else {
        std::cout << "UNKNOWN LOSS TYPE: " << loss_name << std::endl;
        exit(1);
    }

    AT_DISPATCH_FLOATING_TYPES(sigma.scalar_type(), "render_cuda", ([&] {
                render_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    origin.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    gt_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    grad_sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    loss_type);
            }));

    cudaDeviceSynchronize();


    return {pred_dist, gt_dist, grad_sigma};
}


// origin: N x T x 3, points: N x M x 3, tindex: N x M; returns occupancy: N x T x H x L x W
torch::Tensor init_cuda(
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid) {

    const auto N = points.size(0);
    const auto M = points.size(1);

    const auto T = grid[0];
    const auto H = grid[1];
    const auto L = grid[2];
    const auto W = grid[3];

    const auto dtype = points.dtype();
    const auto device = points.device();
    const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
    auto occupancy = torch::zeros({N, T, H, L, W}, options);

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "init_cuda", ([&] {
                init_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    occupancy.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
            }));

    cudaDeviceSynchronize();

    return occupancy;
}