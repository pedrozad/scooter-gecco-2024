import random
import datetime
import time

import multiprocessing
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# NOT IMPLEMENTED import moead # to have moead

import argparse

import copy
import geopandas as gdp
import osmnx as ox
import networkx as nx
import pandas as pd
from pathlib import Path



SCENARIO = "Malaga-Subway"
MAP_INPUT_FILE = "map-Malaga-Subway-all--scooter-walking-subway--nearest-path-ONLY-CYCLEWAY-wBUS-wMetro"

# CONSTANT_COST = 0.00048 # Millions of euros per metre

# MIN_COST = 0
# MAX_COST = 1171337.95399999 #* CONSTANT_COST
# MIN_TIME = 1393.04855117
# MAX_TIME = 1745.86999262

# EVO* CONSTANTS
# MIN_COST = 0
# MAX_COST = 125704.01099999978 # * CONSTANT_COST ( aprox 60,337,925.27 €)
# MIN_TIME = 1393.048551165003
# MAX_TIME = 1745.8699926158945

# New Map CONSTANTS
MIN_COST = 0
MAX_COST = 125869.75899999992 # * CONSTANT_COST ( aprox 60,337,925.27 €)
MIN_TIME = 1392.8303198078484
MAX_TIME = 1715.5414956736442

# MIN_RATIO = 0.017028428700404805
# MAX_RATIO = 0.05255499331895058 
MIN_RATIO = 0
MAX_RATIO = 1


# # I/O Configuration
import sys
IN_COLAB = 'google.colab' in sys.modules

base_path = ''
# if IN_COLAB:
#     from google.colab import drive
#     drive.mount('/content/gdrive')
#     base_path = '/content/gdrive/Shareddrives/happy_mob'
#     get_ipython().system('ls /content/gdrive/Shareddrives/happy_mob/')
# else:

base_path = '.'

data_path = base_path + '/data-osm/'
data_path = data_path + SCENARIO + '/'


results_path = data_path + "results-multiobjective/"
images = results_path + "images/"
map_path = data_path + "/maps/"

Path(results_path).mkdir(parents=True, exist_ok=True)
Path(images).mkdir(parents=True, exist_ok=True)

Path(data_path).mkdir(parents=True, exist_ok=True)
Path(map_path).mkdir(parents=True, exist_ok=True)



TIME_WEIGHT = "time_experiment"
# # Parameter of Algorithms



# # To initilize the toolbox


def init_opti(algorithm_choice):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0) ) # Objetives time vs cost
    creator.create("Individual", list, typecode='d', fitness=creator.FitnessMin)

    toolbox.register("individual", init_ind, icls=creator.Individual, ranges=range(0,LEN_SEARCH_SPACE))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluation)
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutUniformInt, low=[x[0] for x in BOUNDS],
    #                 up=[x[1] for x in BOUNDS], indpb=0.1 / NDIM)
    if(algorithm_choice == "SPEA2"):
        toolbox.register("select", tools.selSPEA2)
        # SPEA2
    elif(algorithm_choice == "RS"):
        # toolbox.register("select", tools.selRandom)
        toolbox.register("select", tools.selBest, fit_attr='fitness.values')
        # Random Search
    else:
        toolbox.register("select", tools.selNSGA2)
        # NSGA2 (individuals, k, nd= Specify the non-dominated algorithm stanrd default)")
        # MOEAD also uses the non-dominated selection
    


    # One MAX
    # toolbox.register("evaluate", evalOneMax)
    # toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUFLIP)
    #toolbox.register("select", tools.selTournament, tournsize=3)


# # Initialization of individuals and populations
def init_ind(icls, ranges):
    genome = list()

    for p in ranges:
        #genome.append(0)
        genome.append(np.random.randint(0,2))
    return icls(genome)


# # Fitness evaluation

def get_distances(map_graph, sections, schools, weight_metric="time_w_c_m"):
    """
        Get Distances in a list
    """    
    distances_weight = list()
    
    for orig_node in sections:
        for dest_node  in schools:
            # print('\tSchools to process: {}'.format(schools_to_process))
            distance, _ = get_route_and_distance(map_graph, orig_node, dest_node, weight_metric)
            distances_weight.append(distance)
            
    return distances_weight



def pair_to_list_tuples(pair_list):
    list_tuples = []
    sections = set()
    schools = set()

    for i in pair_list.itertuples():
        school = getattr(i,"schools")
        section = getattr(i,"sections")

        schools.add(school)
        sections.add(section)

        list_tuples.append( (school, section) )

    return list_tuples



# ## Some useful functions

# Function defined to get the route and the distance between two points using Dijkstra
def get_route_and_distance(map_graph, origin_id, dest_id, weight_metric):
    '''
        @param map_graph : Graph
        @param origin_id: source, Node id of origin
        @param dest_id: target, Node id of destination
        @param weight_metric : String, Function or None, that represent the objective to minimize
            Function should be (\\u,v,data -> number)

        @return distance: distance [in weight] of the route
        @return path : list of nodes from source to target
    '''

    return  nx.bidirectional_dijkstra(map_graph, origin_id, dest_id, weight_metric)

def get_distances_pair(map_graph, pair_list, weight_metric=TIME_WEIGHT, verbose=False):
    """
        Function return the distances of serie of O-D points in a list of tuples

        If verbose is True:
          @return Dataframe of the distances [in weight] of each route
          @return Dataframe of the osmid edges that makes each route
          @return String of pair (origin,destination) that causes errors
          @return Number of routes processed (orign x destination)
          @return Dataframe of the osmid edges that makes each route
          @return Dataframe of the type of edges (follow the prioritize list) taken in each routes
        else:
           @return List of the distances [in weight] for each pair
           @return List of metric (change_mode_e-scooter-walking/number_edges-e-scooter-walking) for each pair

        @param map_graph  : Graph to search the routes
        @param pair_list  : datframe, where schools are the node origns and sections are the destination node
        @param weight_metric   : String or Function (u,v,data)-> number to represent the weight
        @param verbose : Boolean for more information, default False
    """


    # if(verbose):
        # routes_to_proccess = len(pair_list)
        # districts_to_process = len(sections)
        # print('Routes to process: {}'.format(routes_to_proccess))
        # route_type_dict = dict() - LEGACY
        # route_osmid_edges_dict = dict()  - LEGACY
        # times_routes_dict  = dict() - LEGACY
        # route_type_dict['dest'] = "time" - LEGACY

        # Add the destination
        # route_osmid_edges_dict['dest'] = "time"  - LEGACY
        # times_routes_dict['dest'] = "time" - LEGACY
        # total_orign_target_viewed = 0 - LEGACY
        # error = "orig_node,dest_node\n"
        # distances_dict = dict() - LEGACY
        # distances_dict['dest']= "time" - LEGACY



    distances_weight = list() # Save the total weight of the shortest route
    
    list_ratio_change_mode_number_edges = list()  # Save the list of change_mode_active/# edges #NEEDED
    list_edges_that_changed = list() # Save the list of all edges of e-scooter to obtain the cost #NEEDED
    

    for i in pair_list:
        # if verbose:
        #     # distances_weight = list() # Save the total weight of the shortest route - LEGACAY
        #     # LEGACY
        #     # route_type_list = list() # Save how the routes are compound
        #     # times_routes = list() # Save the tuple of how many times walk, scooter, and total
        #     # route_osmid_edges_list = list()

        #     route_osmid_edges = ""

        orig_node = i[0]
        dest_node = i[1]
        try:
            # Map should be a directed graph with weight in non-negative integer
            distance, route_of_nodes = get_route_and_distance(map_graph,orig_node,dest_node, weight_metric) # get distance and routes
            route_edges_attr = ox.utils_graph.route_to_gdf(map_graph, route_of_nodes,weight=weight_metric) # list of Edges attributes of route

            # Save the time
            distances_weight.append(distance)




            #######################################################
            ## CALCULATE THE CHANGE e-scoooter-walking / EDGES   ##
            #  AND GET THE U,V,K of those edges that changed     ##
            #######################################################
            # Calculated the list of changes/#edges_walk+scooter
            is_before_escooter = False
            is_before_walk = False
            change_mode_walk_escooter = 0 # If equal to 0, then only was bus or subway or (only one way)
            times_scooter = 0
            times_walk = 0
            # if(verbose):
            #     route_osmid_edges = '-' # OSMID Route

            for edge in route_edges_attr.itertuples():

                if (verbose):
                    route_osmid_edges = route_osmid_edges  + '{}-'.format(getattr(edge, 'osmid'))

                edge_walk = getattr(edge, 'edge_walk', None)
                edge_scooter  = getattr(edge, 'edge_scooter', None)
                edge_subway = getattr(edge, 'edge_subway', None)
                edge_bus = getattr(edge, 'edge_bus', None)
                edge_changed = getattr(edge, 'edge_changed', None)



                if bool(edge_subway) or bool(edge_bus):
                    is_before_escooter = False
                    is_before_walk = False
                    # if(verbose):
                    #     print(f"edge subwway={bool(edge_subway)} and edge_bus={bool(edge_bus)}")

                elif bool(edge_scooter) or bool(edge_changed):
                    times_scooter += 1
                    is_before_escooter = True # Now, I am using the e-scooter
                    if is_before_walk:
                        # if(verbose):
                        #     print("\t********before was walking?, then change")

                        change_mode_walk_escooter +=1
                    is_before_walk = False
                    # if(verbose):
                    #     print(f"e-scooter")

                elif bool(edge_walk):
                    times_walk += 1
                    is_before_walk = True # Now, I am walking
                    if is_before_escooter:

                        # if(verbose):
                        #     print("\t********before was e-scooter?, then change")
                        change_mode_walk_escooter += 1
                    is_before_escooter = False

                    # if(verbose):
                    #     print(f"walk")
                else:
                    # if(verbose):
                    #     print("********************")
                    #     print("ERROR")
                    #     print(f"edge_walk = {edge_walk}; edge_scooter  = {edge_scooter}")
                    #     print(f"edge_subway = {edge_subway}; edge_bus = {edge_bus}")
                    #     print(f"edge_drive = {edge_changed}")
                    #     print(edge)
                    #     print("**********************")
                    pass

                # End for each edge of the route attr
                # if(verbose):
                #     print("-###################################-")
                #     print("-###################################-")
                #     print(f"Change mode e-scooter/walking {change_mode_walk_escooter}")
                #     print(f"Number of edges e-scooter or walking {times_scooter + times_walk}")
                #     print(route_osmid_edges)

                ratio_change_mode_edges_e_w = change_mode_walk_escooter/(times_scooter+times_walk)

                list_ratio_change_mode_number_edges.append(ratio_change_mode_edges_e_w)
                # End the calcutaion of the ratio (inside for)
                # ----------------------------------------------- #
                # (Still inside for)
                # Start to obtain the u,v,k of those edges that changed

                ss_boolean = getattr(edge,"search_space",None)
                edge_change = getattr(edge,"edge_changed",None)

                if( bool(ss_boolean) and bool(edge_change)):
                    u_escooter,v_escooter,k_escooter = getattr(edge,"Index")
                    u_v_k_escooter = (u_escooter,v_escooter,k_escooter)
                    if not(u_v_k_escooter in list_edges_that_changed): # Saving time
                        list_edges_that_changed.append(u_v_k_escooter)

                # LEGACY
                # if(verbose):
                #     len_all_edges_route += len(route_edges_attr)


            #######################################################
            ##      END CALCULATING RATIO AND LIST_EDGES         ##
            #######################################################


        except:
            # if(verbose):
            #     error_message = f'{orig_node},{dest_node}\n'
            #     error += error_message
            #     print(error_message)
            pass


        ###############################
        ## This should be outside the try/except because
        ## All (array=DF) must be of the same length
        ###############################





        # LEGACY CODE
        # if(verbose):
        #     route_type_list.append(route_type)
        #     # length_distances.append(length_distance)
        #     # try:
        #         # times_routes.append( (times_walk, times_scooter, route_size))
        #         # except:
        #         # print("There is a problem with route_size because the route doesn't exist")
        #     route_osmid_edges_list.append(route_osmid_edges)

        #     distances_dict[i] = distances_weight
        #     route_type_dict[i] = route_type_list
        #     # length_distances_dict[i] = length_distances
        #     # times_routes_dict[i] = times_routes
        #     route_osmid_edges_dict[i] = route_osmid_edges_list

        #     routes_to_proccess -= 1
        #     # print('Routes to process: {}'.format(routes_to_proccess))

    # break


    # End for each orign nod in section

    # LEGACY CODE
    # if(verbose):
    #       return pd.DataFrame(distances_dict), pd.DataFrame(route_osmid_edges_dict), \
    #                     error, total_orign_target_viewed, \
    #                     pd.DataFrame(route_type_dict)
    # else:
    #        return distances_weight, list_fc


    return distances_weight, list_ratio_change_mode_number_edges, list_edges_that_changed



DISTRICT_DATA_FILE = f'districts-{SCENARIO}-data-with-nodes.csv'


districts_data_df = pd.read_csv(data_path + DISTRICT_DATA_FILE)
sections = districts_data_df['node'].tolist()
len(sections)

filename_pair_list="pair_less_than_3600_new_points.csv"
pair_list_df = pd.read_csv(data_path+filename_pair_list)
pair_list_df_selected = pair_list_df.loc[pair_list_df["sections"].isin(sections)]

LIST_PAIR_OD = pair_to_list_tuples(pair_list_df_selected)
print(len(LIST_PAIR_OD))


def get_distances_pair_mean(map_graph, pair_list=LIST_PAIR_OD, weight_metric=TIME_WEIGHT, verbose=False):
    """        Function return the mean of distances of serie of O-D points in a list of tuples
    """
    #list_distances = get_distances(map_graph,schools=schools, sections=sections, weight_metric=TIME_WEIGHT)
    list_distances, list_ratio_change_mode_number_edges, list_edges_that_changed  = get_distances_pair(map_graph, pair_list, weight_metric=TIME_WEIGHT, verbose=False)
    # print(f"list_distances={list_distances}")
    # print(f"list_ratio_change_mode_number_edges={list_ratio_change_mode_number_edges}")
    # print(f"list_edges_that_changed={list_edges_that_changed}")
    to_return_mean_distances = np.mean(list_distances)
    to_return_mean_ratio = np.mean(list_ratio_change_mode_number_edges)
    
    return to_return_mean_distances, to_return_mean_ratio, list_edges_that_changed


# # Evaluation function (Fitness)
def get_cost(individual,list_edges_that_changed,evostarcost):
    cost = 0
    if not(evostarcost):
        # indv_intersect = [0 for i in range(LEN_SEARCH_SPACE)]
        unique = set(list_edges_that_changed)
        for u,v,k in unique:
            pos = dict_of_search_space_uvk_to_pos[(u,v,k)]
            if(individual[pos]):
                cost_temp = dict_of_search_space_cost[pos]
                if(cost_temp <=0):
                    print("There is something bad here!!!")
                cost += cost_temp
                
    else:
        for i in range(0,len(individual)):
            if(individual[i]):
                cost_temp = dict_of_search_space_cost[i]
                if(cost_temp <=0):
                    print("There is something bad here!!!")
                cost += cost_temp
    
    
    return cost


def evaluation(individual):
    #start_time_eva = time.time() 
    
    bounded = copy.deepcopy(G)
    # #Copio cada vez un grafo distinto
    # # len(individual) == LEN_SEARCH_SPACE
    for i in range(0,len(individual)):
        if(individual[i]):
            u,v,k = dict_of_search_space_u_v[i]
            data = bounded.get_edge_data(u, v, key=k)
            if(data == None):
                print("We have a situation here")
            else:
                data["time_w_c"] = data["time_experiment_new"]
                data[TIME_WEIGHT] = data["time_experiment_new"]
                data["edge_changed"] = True 

    objective1, objective3, list_edges_that_changed = get_distances_pair_mean(map_graph=bounded, weight_metric=TIME_WEIGHT)

    objective1 = (objective1-MIN_TIME)/(MAX_TIME-MIN_TIME)
   
    
    objective2 = get_cost(individual,list_edges_that_changed,evostarcost=False) #* CONSTANT_COST
    objective2 = ((objective2-MIN_COST)/(MAX_COST-MIN_COST))
 
    objective3 = (objective3-MIN_RATIO)/ (MAX_RATIO - MIN_RATIO)
 
    # print(f"{np.sum(individual)}->({objective1}, {objective2})")
    #print('\t\t\tOne evaluation in POB {} NGEN{} in seconds {} CXPB {} MUFLIP {} CPUS {}'.format(MU,NGEN,float(end_time_eva-start_time_eva),CXPB, MUFLIP,cpus))
    return objective1, objective2, objective3

# Saving the population information to csv
def saving_pop_dict(pop, counter_i, file_path, show_ind=True, is_append=True):
    """"
        Save the population dict to a csv indicated in file_path
        
        @param pop Population:  Toolbox population
        @param counter_i : Number. Number of generation
        @param file_path : String. File_path to save the csv
        @param show_ind  : Boolean. True default, to save the individual
        @param is_append : Boolean. True default, Mode append to the csv and header= False index=False
    """
    pop_dict = dict()
    pop_dict["gen"] = list ()
    pop_dict["individual"]  = list ()
    pop_dict["fitness1"] = list ()
    pop_dict["fitness2"] = list ()
    pop_dict["fitness3"] = list ()
    
    for ind in pop:
        # print(f"ind.fitness-->{ind.fitness}<---#")
        pop_dict["gen"].append(counter_i)
        if(show_ind):
            pop_dict["individual"].append(ind)
        pop_dict["fitness1"].append(ind.fitness.values[0])
        pop_dict["fitness2"].append(ind.fitness.values[1])
        pop_dict["fitness3"].append(ind.fitness.values[2])
    
    if(show_ind):
        pop_df = pd.DataFrame(data=pop_dict, columns=["gen", "individual", "fitness1", "fitness2","fitness3"])
    else:
        pop_df = pd.DataFrame(data=pop_dict, columns=["gen", "fitness1", "fitness2", "fitness3"])

    # SAVE DATA
    if(is_append):
        pop_df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        pop_df.to_csv(file_path, mode='w', index=False)
                
# # MAIN
def main(file_path, show_ind, MU, show_pareto=False, algorithm_choice="NSGA2",df_new_individuals=pd.DataFrame()):

    pareto = tools.ParetoFront()

    pop = toolbox.population(n=MU)
    counter = 0
    if (algorithm_choice != "RS"):
        for indv in df_new_individuals.itertuples():
            new_ind = eval(getattr(indv,"individuals"))
            pop[counter] = creator.Individual(new_ind)
            counter += 1
            # print(np.sum(new_ind))
    #else: # (algorithm_choice == "RS"): 
    #   Do not update population 
    graph = []
    data = []

    counter_i = 0

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    
    
    print("Inicio del Main")
    if (algorithm_choice == "MOEAD"):
        # maxEvaluations=NGEN,
        ea = moead.MOEAD(pop, toolbox, MU, CXPB, MUFLIP, ngen=NGEN,
                          stats=stats, halloffame=pareto, nr=NR, delta=DELTA, T=SIZE_NEIGHBORHOOD)
        pop, pareto, graph, data = ea.execute_new(file_path=file_path, show_ind=show_ind, show_pareto=show_pareto,df_new_individuals=df_new_individuals)
    elif (algorithm_choice == "RS"):
        
        # First evaluation
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                graph.append(ind.fitness.values)
        # pop = toolbox.select(pop, len(pop))
        # SAVING POPULATION DATA
        saving_pop_dict(pop, counter_i, file_path, show_ind=show_ind, is_append=False)
        pop_all = copy.deepcopy(pop)
        
        # Begin the generational process
        for gen in range(1, NGEN):
            new_pop = toolbox.population(n=MU)
            
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in new_pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                   
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                graph.append(ind.fitness.values)

            pop = new_pop
            # pop = toolbox.select(new_pop, len(pop)) # Not selection
            data.append(fitnesses)
            pop_all += pop

            # Print STATS DATA
            record = stats.compile(pop)
            print(counter_i,record)
            counter_i += 1
            saving_pop_dict(pop, counter_i, file_path, show_ind=show_ind, is_append=True)
                    
            pareto.update(pop)
            
            #end_time = time.time()
            #print('\nProcessed {} ind in GEN {} in {} seconds.'.format(MU,gen,float(end_time-start_time)))
        
        pareto.update(pop_all)

        # Print STATS DATA
        record = stats.compile(pop)
        print(counter_i,record)
        counter_i += 1
        
        
        saving_pop_dict(pop, counter_i, file_path, show_ind=show_ind, is_append=True)
        if(show_pareto):
            pareto_file = file_path.split(".csv")[0] + "_show_solution.csv"
            saving_pop_dict(pareto, counter_i, pareto_file, show_ind=True, is_append=False)
        
        pass
    
    else:
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        
        
        #start_time = time.time()
        #print(f'Empece el GEN 0 a las {time.strftime("%H:%M:%S %d-%m-%Y",  time.gmtime(start_time))}')
        #print("PreFitness 1")
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        #print("PostFitness 1")
        
        #end_time = time.time()
        #print('Processed {} Ind in {} seconds.'.format(MU,float(end_time-start_time)))
        
        data.append(fitnesses)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            graph.append(ind.fitness.values)
        
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done (NSGA2)
        pop = toolbox.select(pop, len(pop))

        # SAVING POPULATION DATA
        saving_pop_dict(pop, counter_i, file_path, show_ind=show_ind, is_append=False)
        
        # Begin the generational process
        for gen in range(1, NGEN):
            #start_time = time.time()
            #print(f'\nEmpece GEN {gen} a las {time.strftime("%H:%M:%S %d-%m-%Y",  time.gmtime(start_time))}')
            # print(f'GEN {gen}')
            # Vary the population
            if(algorithm_choice == "NSGA2"):
                offspring = tools.selTournamentDCD(pop, len(pop))
            else:
                # Use Tournament Tournasize=2
                offspring = tools.selTournament(pop, len(pop),tournsize=2)
            offspring = [toolbox.clone(ind) for ind in offspring]


            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)
                
                # Always mutate
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            data.append(fitnesses)


            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                graph.append(ind.fitness.values)


            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)


            # Print STATS DATA
            record = stats.compile(pop)
            print(counter_i,record)
            counter_i += 1
            saving_pop_dict(pop, counter_i, file_path, show_ind=show_ind, is_append=True)
                    
            pareto.update(pop)
            
            #end_time = time.time()
            #print('\nProcessed {} ind in GEN {} in {} seconds.'.format(MU,gen,float(end_time-start_time)))
        
        pareto.update(pop)

        # Print STATS DATA
        record = stats.compile(pop)
        print(counter_i,record)
        counter_i += 1
        
        
        saving_pop_dict(pop, counter_i, file_path, show_ind=show_ind, is_append=True)
        if(show_pareto):
            pareto_file = file_path.split(".csv")[0] + "_show_solution.csv"
            saving_pop_dict(pareto, counter_i, pareto_file, show_ind=True, is_append=False)

    return pop, pareto, graph, data






if __name__ == "__main__":
           
    """"
        TO PARSE ARGUMENTS
    """
    
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-s","-seed", type=int, help='Seed for random', required=True)
    parser.add_argument("-pm", type=float, help='Mutation probability of changing each bit', required=True)
    parser.add_argument("-pc", type=float, help='Crossover probability', required=True)
    parser.add_argument("-POB","-MU", "-poblacion",  "-pob", type=int, help='Number of individual for population', required=False, default=4)
    parser.add_argument("-GEN","-generacion", "-NGEN","-gen", type=int, help='Number of generations', required=False, default=5)
    parser.add_argument("-CPUS","-cpus","-hilos", type=int, help='Number of cpus', required=False, default=4)
    parser.add_argument("-v","-verbose", "-show_pareto_sol", action='store_true', required=False, help="Show in the last generation the pareto solutions and its individuals, for example [1,1,0,0....]")
    parser.add_argument("-show_ind", action='store_true', required=False, help="Show in the file, the solution individuals [1,1,0,0....]")
    parser.add_argument("-f", "-filepath", required=False, help="Filepath with the filename to upload some individuals. This individuals will be in the first n-position of the population. If number of individuals(n) is greater than -pob will be given error. Otherwise, the difference will be randomly selected")
    
    parser.add_argument("-nr", default=2,
                        required=False, help="MOEAD->nr: Maximal number of individuals replaced by each child")
    parser.add_argument("-delta", default=0.9,
                        required=False, help="MOEAD->delta: the probability that parent solutions are selected from neighbourhoods")
    parser.add_argument("-sn","-size_neighborhood", default=20,
                        required=False, help="MOEAD->size_neighborhood")
    
    parser.add_argument("-algorithm", "-algo", "-a",
                        type=str,
                        choices=['NSGA2', 'SPEA2', 'nsga2', 'spea2','MOEAD','moead', 'RS', 'rs'],
                        default='NSGA2', required=False,
                        help='Choose algorithms between options')


    start_time_stamp = time.time() 
    print(f'Started at {time.strftime("%H:%M:%S %d-%m-%Y",  time.gmtime(start_time_stamp))}')
    args = parser.parse_args()
    arguments = vars(args)

    arguments
    print(arguments)

    show_pareto = arguments["v"] # Boolean, print pareto_front individual solution
    show_ind = arguments["show_ind"] # Boolean, print each individual solution for generation
    
    # # Search Space
    G_gpkg_nodes = gdp.read_file(map_path + f'{MAP_INPUT_FILE}.gpkg', layer="nodes").set_index("osmid")
    G_gpkg_edges = gdp.read_file(map_path + f'{MAP_INPUT_FILE}.gpkg', layer="edges").set_index(['u', 'v', 'key'])

   
    SEED = arguments["s"] # SEED
    random.seed(SEED)
    df_new_individuals = pd.DataFrame()

    filepath_individuals = arguments["f"]
    if not(filepath_individuals is None):
        df_new_individuals = pd.read_csv(filepath_individuals, header=0)
        if(df_new_individuals.shape[0]>arguments["POB"]):
            print("Number of individuals is greater than population")
            sys.exit
    
        
    G = ox.graph_from_gdfs(G_gpkg_nodes, G_gpkg_edges)


    # ## Formatting graph
    SCOOTER_SPEED = 2.78 # m/sec <- 10 KM/h
    WALKING_SPEED = 1.25 # m/sec <- 4,5 KM/h


    list_drive_weird = []
    num_lanes_total = 0
    count_subway = -1
    value_error_list = []

    num_lanes_great_eq_2 = 0
    dict_of_search_space_osmid = {} # key 0.., value:osmid
    dict_of_search_space_u_v = {} # key 0.., value:(u,v,k)
    dict_of_search_space_cost = {} # key 0.., value:length (cost)
    dict_of_search_space_uvk_to_pos = {} # key (u,v,k) -> value: pos
    
    for u, v, k, data in G.edges(data=True, keys=True):
        try:
            data["edge_walk"] = bool(data["edge_walk"])
            data["edge_scooter"] = bool(data["edge_scooter"])
            data["edge_subway"] = bool(data["edge_subway"])
            data["edge_drive"] = bool(data["edge_drive"])
            
            # Search Space and Edge_changed
            data["search_space"] = False
            data["edge_changed"] = False 
            
            data["time_w"] = float(data["time_w"])
            data["time_w_c"] = float(data["time_w_c"])
            data[TIME_WEIGHT] = float(data[TIME_WEIGHT])
            data["time_w_m"] = float(data["time_w_m"])

            data['length'] = float(data['length'])

            try:
                data["osmid"] = int(data["osmid"])
            except:
                # print("Subway doesn't have osmid because is created")
                data["osmid"] = count_subway
                count_subway -= 1


            if (data["edge_drive"]):
                num_lanes_total += 1
                if ((not data["edge_scooter"])):
                    try:
                        data["lanes"]=int(data["lanes"])
                        if (data["lanes"] >= 2):
                            if(not(data["highway"] in ["motorway", "motorway_link"])):
                                data["time_experiment_new"] = data['length'] / SCOOTER_SPEED
                                data["cost"] = 10
                                dict_of_search_space_uvk_to_pos[(u,v,k)] = num_lanes_great_eq_2
                                dict_of_search_space_osmid[num_lanes_great_eq_2] = data["osmid"]
                                dict_of_search_space_u_v[num_lanes_great_eq_2] = (u,v,k)
                                try:
                                    dict_of_search_space_cost[num_lanes_great_eq_2] = float(data['length'])
                                except:
                                    print("Error en incorporar el length en (u=",u,",v=",v,")")

                                num_lanes_great_eq_2 += 1
                                data["search_space"] = True   # To filter by our search_space
                                # data["edge_changed"] = False  # To have a way to know if it was changed

                    except KeyError:
                        data["drive_weird"] = True
                        # print("drive_weird" + " :" + f'u={u},v={v}')
                        list_drive_weird.append((u,v,k,data))
                    except ValueError:
                        value_error_list.append( (u, v, data["lanes"]))
                        data["lanes"]=0

        except KeyError:
            print(data)
            print(KeyError)
            print("Error in change the type")
            print(f'u={u},v={v}')
            # break


    # ## Search Spa
    print(len(dict_of_search_space_osmid))
    print(len(dict_of_search_space_u_v))

    LEN_SEARCH_SPACE = num_lanes_great_eq_2



    NGEN = arguments["GEN"] # Number of Generation
    MU = arguments["POB"]  # Number of individual in population
    cpus = arguments["CPUS"]
    CXPB = arguments["pc"] #Crossover probability
    MUFLIP = arguments["pm"] # Mutation probability of changing each bit
    
    NR =  arguments["nr"]
    DELTA = arguments["delta"]
    
    SIZE_NEIGHBORHOOD = arguments["sn"]
    MTPB = 1 #Mutation probability
    NDIM = LEN_SEARCH_SPACE # Number of dimension of the individual (=number of gene)


    # INPUT_FILE_SCHOOL = f'csv-{SCENARIO}-schools-input.csv'
    # schools_data_df = pd.read_csv(data_path + INPUT_FILE_SCHOOL)
    # schools = schools_data_df['node'].tolist()
    # len(schools)


    
    
    

    # # Bounds on the first 3 genes
    # LOW1, UP1 = 0, 28
    # # Bounds on the last gene
    # LOW2, UP2 = 0, 5

    # BOUNDS = [(LOW1, UP1) for i in range(NDIM-1)] + [(LOW2, UP2)]


    
    toolbox = base.Toolbox()
    
    algorithm_choice=arguments["algorithm"].upper()
    # cpus = multiprocessing.cpu_count()
    # print("With {} CPUS".format(cpus))
    with multiprocessing.Pool(cpus) as pool:
        
        init_opti(algorithm_choice)

        #  #Multiprocessing pool
        toolbox.register("map", pool.map)
        
        
        
        time_stamp = time.time() 
        filename = '{}-{}-{}-{}-{:01.2f}-{:1.2f}-{}-{}-{}-{}'.format(
            SCENARIO, SEED, NGEN, MU, CXPB, MUFLIP, cpus, algorithm_choice, \
            (time.strftime("%Y-%m-%d_%H-%M-%S",  time.gmtime(time_stamp))),
            sys.platform)
        file_path = results_path + 'record_data-'+filename +".csv"
        
        print(file_path, flush=True)
        
        
        # show_ind = False # To save individual to csv
        
        pop, optimal_front, graph_data, data = main(file_path, show_ind,MU, show_pareto, algorithm_choice, df_new_individuals)
        print("fin de Main\n")
        end_time_stamp = time.time() 
        print('\n\nProcessed population {} in NGEN {} in {} seconds. CXPB {} MUFLIP {} CPUS {} algorithm {}\n'.format(MU,NGEN,float(end_time_stamp-start_time_stamp),CXPB, MUFLIP,cpus,algorithm_choice))
        # Saving the Pareto Front, for further exploitation
        with open(results_path+'pareto_front'+filename+'.txt', 'w') as front:
            for ind in optimal_front:
                front.write(str(ind.fitness) + '\n')

