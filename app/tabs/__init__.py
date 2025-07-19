# tabs package initialization
from app.tabs.images_tab import ImagesTab
from app.tabs.masks_tab import MasksTab
from app.tabs.depth_tab import DepthTab
from app.tabs.features_tab import FeaturesTab
from app.tabs.matching_tab import MatchingTab
from app.tabs.reconstruct_tab import ReconstructTab
from app.tabs.gsplat_tab import GsplatTab

# All available tabs
__all__ = [
    'ImagesTab',
    'MasksTab',
    'DepthTab',
    'FeaturesTab',
    'MatchingTab',
    'ReconstructTab',
    'GsplatTab'
]