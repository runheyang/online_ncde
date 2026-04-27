"""Online NCDE 可视化工具。

主要 API：
    from online_ncde.visualization.occ_renderer import (
        OCC3D_CLASS_NAMES, OCC3D_COLORS,
        find_visible, render_voxel_into_figure,
    )
"""

from online_ncde.visualization.occ_renderer import (  # noqa: F401
    OCC3D_CLASS_NAMES,
    OCC3D_COLORS,
    find_visible,
    render_voxel_into_figure,
    clear_figure,
)
