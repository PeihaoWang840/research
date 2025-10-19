import numpy as np
import matplotlib.pyplot as plt

def roulette_wheel_selection(probabilities):
    """Selects an index based on cumulative probabilities."""
    cum_sum = np.cumsum(probabilities)
    r = np.random.rand()
    return np.where(cum_sum >= r)[0][0]

def initialization2(nSample, nVar, VarMax, VarMin):
    """Initializes new positions within the given bounds."""
    return (VarMax - VarMin) * np.random.rand(nSample, nVar) + VarMin

def ACO(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction):
    VarSize = [1, nVar]
    nSample = 40
    q = 0.5
    zeta = 1
    
    # Create empty individual structure using a dictionary
    class Individual:
        def __init__(self):
            self.Position = None
            self.Cost = None
    
    pop = [Individual() for _ in range(nPop)]
    
    for i in range(nPop):
        pop[i].Position = initialization2(1, nVar, VarMax, VarMin)
        pop[i].Cost = CostFunction(pop[i].Position)
    
    # Convert population to list of dictionaries
    pop_dict = [{'Position': p.Position, 'Cost': p.Cost} for p in pop]
    pop_dict.sort(key=lambda x: x['Cost'])
    
    BestSol = {'Position': pop_dict[0]['Position'], 'Cost': pop_dict[0]['Cost']}
    BestCost = np.zeros(MaxIt)
    
    # Calculate solution weights
    w = 1 / (np.sqrt(2 * np.pi) * q * nPop) * \
        np.exp(-0.5 * ((np.arange(nPop) - 1) / (q * nPop)) ** 2)
    p = w / np.sum(w)
    
    for it in range(MaxIt):
        # Compute means
        s = np.array([ind['Position'] for ind in pop_dict[:nPop]])
        
        # Compute standard deviations
        sigma = np.zeros((nPop, nVar))
        for l in range(nPop):
            D = 0
            for r in range(nPop):
                D += np.abs(s[l] - s[r])
            sigma[l] = zeta * D / (nPop - 1)
        
        # Create new population
        newpop = []
        for t in range(nSample):
            new_pos = np.zeros(VarSize)
            for i in range(nVar):
                l = roulette_wheel_selection(p) + 1  # Adjusting to 1-based index
                new_pos[i] = s[l-1, i] + sigma[l-1, i] * np.random.randn()
            
            newpop.append({'Position': new_pos, 'Cost': CostFunction(new_pos)})
        
        # Merge populations
        merged_pop = pop_dict + newpop
        # Sort by cost
        merged_pop.sort(key=lambda x: x['Cost'])
        pop_dict = merged_pop[:nPop]
        
        # Update best solution
        if pop_dict[0]['Cost'] < BestSol['Cost']:
            BestSol = {'Position': pop_dict[0]['Position'], 'Cost': pop_dict[0]['Cost']}
        
        BestCost[it] = BestSol['Cost']
    
    return BestSol['Cost'], BestSol['Position'], BestCost

# Example usage:
def example_cost_function(x):
    """Example cost function, replace with actual problem's cost function."""
    return x[0]**2 + x[1]**2

nPop = 30
MaxIt = 50
VarMin = np.array([0, 0])
VarMax = np.array([1, 1])
nVar = 2

bestf, bestx, BestCost = ACO(nPop, MaxIt, VarMin, VarMax, nVar, example_cost_function)

print(f"Best cost: {bestf}")
print(f"Best position: {bestx}")

plt.plot(BestCost)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.title('ACO Algorithm Performance')
plt.show()