from tower_planner import TowerPlanner

def geometric_stable(tower):
    tp = TowerPlanner(stability_mode='contains')
    return tp.tower_is_cog_stable(tower)

def pairwise_stable(tower):
    tp = TowerPlanner(stability_mode='contains')
    return tp.tower_is_constructible(tower)

def containment_stable(tower):
    tp = TowerPlanner(stability_mode='contains')
    return tp.tower_is_containment_stable(tower)

def com_stable(tower):
    tp = TowerPlanner(stability_mode='contains')
    return tp.tower_is_stable(tower)

def com_stable_n(n):
    def stable(tower):
        if len(tower) == n:
            return com_stable(tower)
        else:
            return not com_stable(tower)
    return stable



def get_all_hypotheses():
    hypotheses = [geometric_stable, pairwise_stable, containment_stable, com_stable]
    hypotheses += [com_stable_n(n) for n in range(2, 6)]
    return hypotheses