import numpy as np
from constants import PATH_NAMES, PATH_RELATIONSHIP_MATRIX, path_idx, VOTE_WEIGHTS
from models import TitanToPathModel

class Pathstrider:
    def __init__(self, name, titan_affinities, entity_trait="Mortal", titan_to_path_model=None):
        self.name = name
        self._titan_to_path_model = titan_to_path_model if titan_to_path_model is not None else TitanToPathModel()
        self.titan_affinities = np.array(titan_affinities).clip(min=0)
        self.trait = entity_trait
        self.golden_one_tenure = 0
        self.heroic_tendency = 0
        self._path_affinities = None
        
        # score 动态计算，初始值0
        self.score = 0 
        
        self.activity = 0
        self.stability = 0
        
        if self.trait == "Reincarnator" or self.trait == "GoldenOne":
            self.is_titan_form = None
        else:
            self.is_titan_form = None  
        self.held_fire_seeds = set() 

        self.recalculate_concepts()

    @property
    def path_affinities(self):
        if self._path_affinities is None:
            self._path_affinities = self._titan_to_path_model.get_path_affinities(self.titan_affinities)
        return self._path_affinities
    
    @property
    def dominant_path_idx(self):
        return np.argmax(self.path_affinities)
    
    @property
    def purity(self):
        path_affs = self.path_affinities
        if np.sum(path_affs) == 0: return 0
        return np.max(path_affs) / np.sum(path_affs)

    def recalculate_concepts(self, path_distribution=None, cosmic_zeitgeist=None):
        """
        重新计算实体的所有衍生概念，包括新的动态评分。
        - path_distribution: 全局的命途饱和度分布
        - cosmic_zeitgeist: 当前的思潮向量
        """
        if not np.all(np.isfinite(self.titan_affinities)):
            self.titan_affinities = np.ones(len(self.titan_affinities))

        self._path_affinities = None 
        
        self.activity = np.linalg.norm(self.path_affinities[:6]) * 10
        self.stability = np.linalg.norm(self.path_affinities[6:]) * 10
        
        base_potential = (self.activity + self.stability) * (1 + self.purity)

        if cosmic_zeitgeist is not None:
            # 实体命途与思潮的契合度
            contextual_multiplier = 1 + np.dot(self.path_affinities, cosmic_zeitgeist)
        else:
            contextual_multiplier = 1.0

        if path_distribution is not None:
            dominance_penalty_factor = path_distribution[self.dominant_path_idx]
            # 饱和度越高，修正因子越小
            saturation_modifier = 1.0 / (1.0 + 2 * dominance_penalty_factor)
        else:
            saturation_modifier = 1.0

        self.score = base_potential * max(0.1, contextual_multiplier) * saturation_modifier

        self.heroic_tendency = self.activity

    def generate_vote_proposal(self):
        """
        生成该实体的选票，即它所期望的理想思潮。
        """
        vote_vector = np.zeros(len(PATH_NAMES))
        my_dom_idx = self.dominant_path_idx
        
        for i in range(len(PATH_NAMES)):
            if i == my_dom_idx:
                vote_vector[i] = 1.0
                continue

            relationship = PATH_RELATIONSHIP_MATRIX[my_dom_idx, i]
            
            if relationship == "SYNERGY":
                vote_vector[i] = 0.5  
            elif relationship == "MENTORSHIP":
                vote_vector[i] = 0.2  
            elif relationship == "REPULSION":
                vote_vector[i] = -0.7 
            elif relationship == "CLASH":
                vote_vector[i] = -1.0 
        
        e_x = np.exp(vote_vector - np.max(vote_vector))
        return e_x / e_x.sum(axis=0)

    def get_vote_weight(self):
        """
        获取该实体投票权重
        """
        return VOTE_WEIGHTS.get(self.trait, 1.0)


    def internal_purification(self, purity_factor):
        if purity_factor <= 0 or len(self.titan_affinities) < 2: return
        dominant_idx = np.argmax(self.titan_affinities)
        dominant_value = self.titan_affinities[dominant_idx]
        reduction_amount = dominant_value * purity_factor
        mask = np.ones(len(self.titan_affinities), dtype=bool)
        mask[dominant_idx] = False
        self.titan_affinities[mask] -= reduction_amount
        self.titan_affinities = self.titan_affinities.clip(min=0)
        self._path_affinities = None

    def __repr__(self):
        dominant_path_name = PATH_NAMES[self.dominant_path_idx]
        
        tags = []
        if self.trait == "Reincarnator":
            if self.name == "Neikos-0496":
                tags.append("白厄")
            else:
                tags.append("卡厄斯兰那")
        elif self.trait == "GoldenOne":
            tags.append(f"黄金裔(任期:{self.golden_one_tenure})")
        
        if self.is_titan_form:
            tags.append(f"泰坦化身: {self.is_titan_form}")
        
        if self.held_fire_seeds:
            tags.append(f"火种({len(self.held_fire_seeds)})")

        if not tags:
            tag_str = f"'{dominant_path_name}'的追随者"
        else:
            tag_str = ", ".join(tags)

        return f"[{self.name}] <{tag_str}>(评分:{self.score:.2f}|纯:{self.purity:.2f}|最强命途:{dominant_path_name}:{self.path_affinities[self.dominant_path_idx]:.2f})"