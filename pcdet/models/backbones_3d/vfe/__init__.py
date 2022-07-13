from .mean_vfe import MeanVFE
from .drop_voxel_vfe import DropVoxelVFE
from .pillar_vfe import PillarVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'DropVoxelVFE': DropVoxelVFE,
}
