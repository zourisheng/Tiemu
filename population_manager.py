import numpy as np
import random
from collections import Counter

from constants import GREEK_ROOTS, TITAN_NAMES, PATH_NAMES
from entities import Pathstrider
from models import TitanToPathModel

class PopulationManager:
    def __init__(self, existing_names: set, name_to_entity_map: dict, 
                 base_titan_affinities: np.ndarray, mutation_rate: float, 
                 purity_factor: float, golden_one_cap: int, 
                 golden_one_reversion_prob: float, titan_to_path_model: TitanToPathModel,
                 max_affinity_norm: float, min_norm: float, max_norm: float):
        
        self.existing_names = existing_names
        self.name_to_entity_map = name_to_entity_map
        self.base_titan_affinities = base_titan_affinities
        self.mutation_rate = mutation_rate
        self.purity_factor = purity_factor
        self.golden_one_cap = golden_one_cap
        self.golden_one_reversion_prob = golden_one_reversion_prob
        self.titan_to_path_model = titan_to_path_model
        self.max_affinity_norm = max_affinity_norm
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.cosmic_tide_vector = np.zeros(len(TITAN_NAMES)) 

    def generate_unique_name(self):
        while True:
            r1, r2 = random.sample(GREEK_ROOTS, 2)
            num = random.randint(1000, 9999)
            name = f"{r1}{r2}-{num}"
            if name not in self.existing_names:
                self.existing_names.add(name)
                return name

    def create_initial_population(self, num_entities: int, population: list, 
                                  cosmic_zeitgeist: np.ndarray):
        for _ in range(num_entities):
            name = self.generate_unique_name()
            entity = Pathstrider(name, self.base_titan_affinities + np.random.normal(0, self.mutation_rate, self.base_titan_affinities.shape), titan_to_path_model=self.titan_to_path_model)
            self.recalculate_and_normalize_entity(entity, None, None) 
            population.append(entity)
            self.name_to_entity_map[name] = entity
        
    def recalculate_and_normalize_entity(self, entity: Pathstrider, path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray):
        entity.internal_purification(self.purity_factor)
        entity.titan_affinities += self.cosmic_tide_vector
        entity.titan_affinities = entity.titan_affinities.clip(min=0)
        self.normalize_affinities(entity)
        if path_distribution is not None and cosmic_zeitgeist is not None:
            entity.recalculate_concepts(path_distribution, cosmic_zeitgeist)

    def normalize_affinities(self, entity_or_blueprint):
        target = entity_or_blueprint.titan_affinities if isinstance(entity_or_blueprint, Pathstrider) else entity_or_blueprint
        norm = np.linalg.norm(target)
        
        if not np.isfinite(norm) or norm == 0 or norm > self.max_affinity_norm:
            normalized_target = (target / (norm + 1e-9)) * self.max_affinity_norm
            if isinstance(entity_or_blueprint, Pathstrider):
                entity_or_blueprint.titan_affinities = normalized_target
            else: 
                entity_or_blueprint[:] = normalized_target
        return None 

    def get_global_path_distribution(self, population: list):
        if not population:
            return np.ones(len(PATH_NAMES)) / len(PATH_NAMES)
        path_counts = np.zeros(len(PATH_NAMES))
        for p in population:
            path_counts[p.dominant_path_idx] += 1
        distribution = path_counts / len(population)
        return distribution

    def update_golden_ones(self, population: list):
        current_golden_ones = [p for p in population if p.trait == "GoldenOne"]
        is_cap_met = len(current_golden_ones) >= self.golden_one_cap
        reverted_count = 0
        for entity in current_golden_ones:
            if entity.golden_one_tenure >= 2:
                reversion_prob = self.golden_one_reversion_prob * (entity.golden_one_tenure - 1)
                if is_cap_met: reversion_prob *= 2
                if np.random.random() < reversion_prob:
                    entity.trait = "Mortal"
                    entity.golden_one_tenure = 0
                    reverted_count += 1
            else: entity.golden_one_tenure += 1
        eligible_for_promotion = [p for p in population if p.trait == "Mortal"]
        if not eligible_for_promotion: return
        
        eligible_for_promotion.sort(key=lambda p: p.heroic_tendency, reverse=True)
        tendencies = [p.heroic_tendency for p in eligible_for_promotion if np.isfinite(p.heroic_tendency)]
        if not tendencies: return
        
        promotion_threshold = np.percentile(tendencies, 95)
        available_slots = self.golden_one_cap - (len(current_golden_ones) - reverted_count)
        
        for entity in eligible_for_promotion:
            if available_slots <= 0: break
            if entity.heroic_tendency >= promotion_threshold and entity.trait == "Mortal":
                entity.trait = "GoldenOne"
                entity.golden_one_tenure = 1
                available_slots -= 1

    def check_and_replenish_population(self, population: list, population_soft_cap: int, 
                                       aeonic_cycle_mode: bool, reincarnator: Pathstrider, 
                                       cosmic_zeitgeist: np.ndarray):
        if len(population) < population_soft_cap * 0.8:
            num_to_add = int(population_soft_cap - len(population))
            if num_to_add <= 0: return
            
            if aeonic_cycle_mode: 
                blueprint = self.base_titan_affinities * (1.5 * random.uniform(0.1, 1.5))
            else:
                golden_ones = [p for p in population if p.trait == "GoldenOne"]
                if not golden_ones: return
                template_entity = random.choice(golden_ones)
                blueprint = template_entity.titan_affinities
                
            self._add_new_entities(population, num_to_add, blueprint, cosmic_zeitgeist)

    def replenish_population_by_growth(self, population: list, num_to_add: int, cosmic_zeitgeist: np.ndarray):
        """Replenish based on the current blueprint from the guide network."""
        blueprint = self.base_titan_affinities
        self._add_new_entities(population, num_to_add, blueprint, cosmic_zeitgeist)

    def _add_new_entities(self, population: list, num_to_add: int, blueprint: np.ndarray, cosmic_zeitgeist: np.ndarray):
        dist_newborns = self.get_global_path_distribution(population)
        for _ in range(num_to_add):
            name = self.generate_unique_name()
            new_affinities = blueprint + np.random.normal(0, self.mutation_rate, self.base_titan_affinities.shape)
            entity = Pathstrider(name, new_affinities, titan_to_path_model=self.titan_to_path_model)
            self.recalculate_and_normalize_entity(entity, dist_newborns, cosmic_zeitgeist)
            population.append(entity)
            self.name_to_entity_map[name] = entity