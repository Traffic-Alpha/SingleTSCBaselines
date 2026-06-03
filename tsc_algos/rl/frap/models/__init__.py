'''
@Author: WANG Maonan
@Description: RL Models for TSC
'''
from .transformer import TransformerJunctionModel
from .mlp import MLPJunctionModel

__all__ = ['TransformerJunctionModel', 'MLPJunctionModel']
