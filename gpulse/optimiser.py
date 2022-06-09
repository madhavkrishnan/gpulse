# This file contains the GA optimizer
import numpy as np
import random
from gpulse.costfn import cost_sig_noise
from deap import base, tools, creator
from gpulse.deap_tools import mutate, init_ind, cx, snip
import copy

def create_job(inputs=None):
    """
    Parameters
    ----------
    inputs : dict, optional
        dictionary with algorithm parameters to use as input for gpulse.Optimiser.
        If not provided, a default template is generated. Cost function args and kwargs
        can be specified after generation.The first argument of the cost function is
        assumed to be the solution individual by gpulse.Optimiser.

        For example with the default cost_function 'cost_sig_noise', a job can be
        generated as,

        ------------------------------------------------------------------------

        job = create_job()
        job['cfn_kwargs'] = {SIGNAL_PSD : sig_array, NOISE_PSD : noise_array}

        ------------------------------------------------------------------------

        where sig_array and noise_array are the power spectral density arrays
        for the signal and noise.

    Returns
    -------
    params : dict
        dictionary containing algorithm parameters

    A description of each key in the dictionary is given below, starred ones are
    not implemented yet.

    fixed_cycles* : Bool
            Allows the number of CPMG cycles to be held fixed
    fixed_rotation* : Bool
            Allows the X rotation angles to be fixed
    fixed_timing* : Bool
            Allows the interpulse spacing to be fixed
    fixed_shots* : Bool
            Allows the number of shots / trials to estimate the cost function
            to be fixed. By default the number of trials are increased with generation.
    NGEN : int,
        Number of generations to run the algorithm for.
    POP_SIZE : int
        Population size.
    MUTPB : float
        Mutation probability
    CXPB : float
        Crossover probability
    GPB : float
        probability of cycle length change during mutation.
    CPMG_CYCLES : int
                Number of CPMG cycles. If fixed size is false, this is used
                as maximum cycle length when initialising the population.
    X_ROT_MAX : int
                Determines the minimum X rotation as 2pi / X_ROT_MAX
    X_ROT_MIN : int
                Determines the mximum X rotation as 2pi / X_ROT_MIN
    TAU_MIN : int
            Minimum interpulse spacing.
    TAU_MAX : int
            Maximum interpulse spacing.
    MAX_SHOTS : int
                Determines the max number of trails used to estimate the cost function
                with increasing generation.
    cost_function : function,
                    The cost function used  by the algorthm. The first argument must
                    accept a pulse individual of the form [(t_1,θ_1), (t_2,θ_2)... (t_n,θ_n)].
                    By default gpulse.costfn.cost_sig_noise is used.
    cfn_args : tuple
            Positional arguments after the pulse individual for the cost function
    cfn_kwargs : dict
            Keyword arguments of the cost function.
    pop_init : function, opt
            function used to initialise the population. Defaults to None if not specified
            and gpulse.Optimiser will use the default init_ind function.
    pinit_args : tuple,
            Positional arguments for the pop_init function.
    optimise : str
            Expects 'max' or 'min' deciding whether to maximise or mimise the cost
            function.

    """

    params = {
              'fixed_cycles'      : False,
              'fixed_rotation'    : False,
              'fixed_timing'      : False,
              'fixed_shots'       : False,
              'NGEN'              : 50,
              'POP_SIZE'          : 100,
              'MUTPB'             : 0.5,
              'CXPB'              : 0.3,
              'GPB'               : 0.5,
              'CPMG_CYCLES'       : 5,
              'X_ROT_MAX'         : 10,
              'X_ROT_MIN'         : 1,
              'TAU_MIN'           : 1,
              'TAU_MAX'           : 10,
              'MAX_SHOTS'         : 1000,
              'cost_function'     : cost_sig_noise,
              'cfn_args'          : [],
              'cfn_kwargs'        : [],
              'pop_init'          : None,
              'pinit_args'        : None,
              'optimise'          : 'max'
              }
    # Add user defined values
    if not inputs is None:
        for k in inputs.keys():
            params[k] = inputs[k]

    return params




class Optimiser:
    """
    Genetic algorithm based optimiser.

    The optimiser is build using deap. It is initialised with a list of jobs
    """

    def __init__(self, *jobs):
        """
        Parameters
        ----------
        jobs : dict or multiple dicts.
            Contains the job(s) the optimiser must run

        """

        self.jobs = jobs
        self.results = None

    def run(self):
        """
        Executes the optimiser for the job instances in self.jobs.

        Returns
        -------
        result : list or list of lists
            job results, each result contains the logbook of the simulation as
            well as the best solutions found during the run.

        """
        results = []
        count = 1
        for j in self.jobs:

            print('Job ' + str(count))
            print('------')
            count += 1
            # Set up deap functions

            # Create fitness attrubte

            # Book keeping for multiple job runs.
            if hasattr(creator, 'Fitness'):
                del creator.Fitness
            if hasattr(creator, 'Individual'):
                del creator.Individual

            if j['optimise'] == 'max':
                creator.create("Fitness", base.Fitness, weights=(1.0,))
            elif j['optimise'] == 'min':
                creator.create("Fitness", base.Fitness, weights=(-1.0,))

            # Create individual generator
            creator.create("Individual", list, fitness=creator.Fitness)

            # Initialise toolbox
            toolbox = base.Toolbox()

            # Register the cost function
            toolbox.register("evaluate",
                            j['cost_function'],
                            *j['cfn_args'], **j['cfn_kwargs']
                            )
            # Register a slection function with the Toolbox
            toolbox.register("select", tools.selTournament, tournsize=5)

            # Register the mutation function

            toolbox.register("mutate",
                            mutate,
                            tau_lims = [j['TAU_MIN'], j['TAU_MAX']],
                            rot_lims = [j['X_ROT_MIN'], j['X_ROT_MAX']],
                            indpb=0.5, vary_length = not j['fixed_cycles'], GPB=j['GPB'])

            # Register a crossover function
            toolbox.register("mate", cx)

            # If a population initialisation function has been passed to the
            # optimiser, it will use it instead of the default one deap_tools.init_ind
            if j['pop_init'] is not None:
                pop_init = j['pop_init']
                init_args = j['pinit_args']
            else:
                # Default initialiser
                pop_init = init_ind
                init_args = ([j['TAU_MIN'], j['TAU_MAX']], [j['X_ROT_MIN'], j['X_ROT_MAX']])

            # Register generator for individual
            toolbox.register("attr_int",
                            pop_init,
                            *init_args
                            )

            toolbox.register("individual", tools.initRepeat,
            creator.Individual, toolbox.attr_int, n=j['CPMG_CYCLES'])

            # Define function to build a population
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # list decides number of shots with increasing generation, when using
            # the filter function, this decides the number of Poisson trials.
            shots = list(map(int, np.linspace(1, j['MAX_SHOTS'], j['NGEN'] + 1)))

            # Build a population
            pop = toolbox.population(n=j['POP_SIZE'])

            # If variable cycles, randomise population cycle length
            if not j['fixed_cycles']:
                pop = list(toolbox.map(snip, pop))

            # Evaluate the fitness of the population.
            fitness = [toolbox.evaluate(ind, shots=shots[0]) for ind in pop]
            for ind, fit in zip(pop, fitness):
                ind.fitness.values = fit

            # Define a statistics class to capture population statistics
            stats = tools.Statistics(key=lambda ind: ind.fitness.values)

            # Register stat functions
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)


            # Get the stats of the current population
            record = stats.compile(pop)

            # Create a log object
            logbook = tools.Logbook()
            logbook.record(gen=0, **record)
            logbook.header = "gen", "max", "avg", 'min', 'std'

            # Create class to track best individual
            best_ind = tools.HallOfFame(1)
            best_ind.update(pop)


            # Run genetic aglorithm over NGEN generations
            # print output headers
            print("Generation \t\t Best Individual" )
            print('\t\t      τ1 τ2..τn | θ1 θ2..θn')
            print('------'*10)
            for g in range(1, j['NGEN'] ):

                # Select offspring from current population
                offspring = toolbox.select(pop, len(pop)-1)

                #clone new individuals from the selection
                offspring = list(map(toolbox.clone, offspring))

                # Apply crossover by pairing adjacent individuals
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < j['CXPB']:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Apply mutation
                for mutant in offspring:
                    if random.random() < j['MUTPB']:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Keep the best individual (elitist strategy)
                offspring.append(best_ind[0])

                # Re-evaluate fitness of all individuals
                fitnesses = [toolbox.evaluate(ind, shots=shots[g]) for ind in offspring]

                # Update fitness
                for ind, fit in zip(offspring, fitnesses):
                    # if fit == 0:
                    #     print("Zero fit")
                    #     break
                    del ind.fitness.values
                    ind.fitness.values = fit

                # Update generation.
                pop[:] = offspring

                # Record fitness statistics for new generation.
                record = stats.compile(pop)
                logbook.record(gen=g, **record )

                # Update best ind
                best_ind.update(pop)

                # print output every NGEN / 10 generations.
                if (g % (j['NGEN']/10) == 0) or (g == j['NGEN']-1):
                    fid = [ind.fitness.values[0] for ind in pop]
                    tau_str = [str(best_ind[0][i][0]) for i in range(len(best_ind[0]))]
                    rot_str = [ '2π/' + str(best_ind[0][i][1]) for i in range(len(best_ind[0]))]

                    best_formated = " ".join(tau_str) + ' | ' + " ".join(rot_str)
                    print(f'{g} \t\t {best_formated} ')
            print('\n')
            results.append([logbook, best_ind[0][:]])

        self.results = copy.deepcopy(results)
        return results
