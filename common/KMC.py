
import networkx as nx
import random
import math




#essential layout and plan
#use an arbitary networkx graph
#this graph will have edges with specified rates
#maybe for future extendablity this should take an update
#rates method which is system dependent
#this graph must have a defined rate between edges
#and a maximum occupany (max_occ) at each node
#similarly each node should have a current occupancy (occ)
#infinit sources and sinks are represented by finite sources and
#sinks with very large max_occ 
#additionally the graph should be allowed to be directed
#this allows us to model the simple exclusion porcess
#https://aiichironakano.github.io/phys516/KMCet.pdf

#kinetic monte carlo based on 
#https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo
class KMC():
    def __init__(self,graph,initial_pos_options={"mode":"random","N":1},
                 convergence_criteria=None,supressErrors=set(),
                 seed=0) -> None:
        self.graph=graph
        self.convergence_criteria=convergence_criteria
        self.initial_pos_options=initial_pos_options
        self.initial_pos=set()
        self.occupied_sites=set()
        self.supressErrors=supressErrors
        self.seed=seed
        random.seed(seed)
        self.validate_graph()
        self.validate_convergence_params()
        
        self.time=0
        self.steps=0
        
    def __rand(self):
        #on the range [0,1)
        r=random.random()
        #on the range (0,1]
        return 1-r
        
    def find_particles(self):
        particles=set()
        for n in self.graph.nodes:
            if(graph.nodes[n]["occ"]>0):
                particles.add(n)
        return particles
        
    def initialize_positions(self):
        if("mode" not in self.initial_pos_options):
            raise Exception("ERROR:Initiallization method required as "
                            +'initial_pos_options={"mode":str}')
        #assumes that particles are already initiallized in the graph which will
        #often be desirable
        done_modes=[None,"done","already_set"]
        if self.initial_pos_options["mode"] in done_modes:
            return
        if("mode" not in self.initial_pos_options):
            raise Exception("ERROR:Selected initialization mode "+self.initial_pos_options["mode"]+
                            " required the number of particles specified as "
                            +'initial_pos_options={"N":int}')
        
        
        if self.initial_pos_options["mode"]=="random":
            warnlvl=1e5
            n=self.initial_pos_options["N"]
            i=0
            total_attempts=0
            nodeslist=[n for n in self.graph.nodes]
            while(i<n):
                r=random.choice(nodeslist)
                if(self.graph.nodes[r]["occ"]<self.graph.nodes[r]["max_occ"]):
                    self.graph.nodes[r]["occ"]+=1
                    i+=1
                total_attempts+=1
                if(total_attempts>warnlvl):
                    print("WARNING: EXCEEDED "+str(warnlvl)+" attempts to place particles. This is likley due to"
                          +" a particle count that is very near to the total allowed number of particles. ")
        self.initial_pos=self.find_particles()
        self.occupied_sites=self.initial_pos
        return self.initial_pos
                
        
    def validate_convergence_params(self):
        supported_criteria=["maxsteps"]
        if(type(self.convergence_criteria)!=dict):
            raise Exception("ERROR: convergence_critera must be a dict not "
                            +str(type(self.convergence_criteria)))
        keys=[k for k in self.convergence_criteria]
        if(len(keys)>1):
            raise Exception("ERROR: Too many convergence criteeria supplied. The dict should have only one key")
        if(keys[0] not in supported_criteria):
            raise Exception("ERROR: criteria "+keys[0]+" is not currently supported for this method. Supported"
                           +" criteria are: "+str(supported_criteria))
        if(keys[0]=="maxsteps"):
            if(type(self.convergence_criteria["maxsteps"])!=int or self.convergence_criteria["maxsteps"]<1):
                raise Exception("ERROR: for selected criteria convergence_criteria must be a dict of the form"
                                +"{maxsteps:x} where x is a positive integer")
        
    def validate_graph(self):
        req_node_keys=set(["occ","max_occ"])
        req_edge_keys=set(["rate"])
        for n in self.graph.nodes:
            for k in self.graph.nodes[n]:
                if(k not in req_node_keys and "unknownNodeKey" not in self.supressErrors):
                    raise Exception("ERROR:UnknownNodeKey '"+k+"' for node "+str(n)+" and will not be used. If this is a mistake"
                                    +" and you know what you are doing then add 'unknownNodeKey' to"
                                    +" supressErrors")
            for k in req_node_keys:
                if(k not in self.graph.nodes[n]):
                    raise Exception("ERROR:Missing required key '"+k+"' for node "+str(n))
        for edg in self.graph.edges:
            for k in self.graph.edges[edg]:
                if(k not in req_edge_keys and "unknownEdgeKey" not in self.supressErrors):
                    raise Exception("ERROR:UnknownNodeKey '"+k+"' for node "+str(n)+" and will not be used. If this is a mistake"
                                    +" and you know what you are doing then add 'unknownEdgeKey' to"
                                    +" supressErrors")
            for k in req_edge_keys:
                if(k not in self.graph.edges[edg]):
                    raise Exception("ERROR:Missing required key '"+k+"' for edge "+str(n))

    def get_rates(self):
        #cosnider transitions from v to j
        #must consider all for digraph
        rates={}
        for v in self.occupied_sites:
            for j in self.graph.neighbors(v):
                #if site j has room for more particles than allow the transition
                if(j["occ"]<j["max_occ"]):
                    rates[(v,j)]=self.graph.edges[v,j]["rate"]
        return rates
    
    def construct_cumulative(self):
        #cosnider transitions from v to j
        #must consider all for digraph
        cumulative_edg_map={}
        cumulative=[]
        for v in self.occupied_sites:
            for j in self.graph.neighbors(v):
                #if site j has room for more particles than allow the transition
                if(self.graph.nodes[j]["occ"]<self.graph.nodes[j]["max_occ"]):
                    rate=self.graph.edges[v,j]["rate"]
                    cumulative_edg_map[len(cumulative)]=(v,j)
                    if(len(cumulative)<1):
                        cumulative.append(rate)
                    else:
                       cumulative.append(cumulative[-1]+rate)
        return cumulative,cumulative_edg_map
            
    def select_rate(self,cumulative):
        low=0
        high=len(cumulative)-1
        r_rate=self.__rand()
        rate=r_rate*cumulative[-1]
        while low <= high:
            mid = low + (high - low) // 2
            # Check if x is present at mid
            if (mid-1<0 and rate<=cumulative[mid]) or (rate>cumulative[mid-1] and rate<=cumulative[mid]):
                return mid
            # If x is greater, ignore left half
            elif cumulative[mid] < rate:
                low = mid + 1
            # If x is smaller, ignore right half
            else:
                high = mid - 1
        # If we reach here, then the element
        # was not present
        return -1   
    
    #transition from v to j
    def do_transition(self,v,j):
        if(self.graph.nodes[v]["occ"]<1):
            raise Exception("ERROR: Site "+str(v)+" has in sufficient particles"
                            +" to transition to "+str(j))
        if(self.graph.nodes[j]["occ"]>=self.graph.nodes[j]["max_occ"]):
            raise Exception("ERROR: Site "+str(j)+" has in too many particles"
                            +" to accept a transition from "+str(v))
        self.graph.nodes[v]["occ"]-=1
        self.graph.nodes[j]["occ"]+=1
        
        #remove old node if empty
        if(self.graph.nodes[v]["occ"]<1):
            self.occupied_sites.remove(v)
        #add new node
        self.occupied_sites.add(j)
        
    def run_conv_checker(self):
        if("maxsteps" in self.convergence_criteria):
            return self.steps >= self.convergence_criteria["maxsteps"]
    def KMC_step(self):
        should_continue=True
        #select and carrier out a transition
        cumulative,transition_mapping=self.construct_cumulative()
        if(len(cumulative)<1):
            print("ERROR: All particles have reached a sink. Please validate that this behavior is expected."
                  +" This commonly happens in diGraphs and when a graph is not properly connected."
                  +"This is some time expected behavior. Please verify. THE SIMULATION HAS NOT "
                  +"REACHED CONVERGENCE AND IS STOPPING EARLY!")
            should_continue=False
            return should_continue
        rate_index=self.select_rate(cumulative)
        transition_nodes=transition_mapping[rate_index]
        self.do_transition(transition_nodes[0],transition_nodes[1])
        r_time=self.__rand()
        delT=(1/cumulative[-1])*math.log(1/r_time)
        self.time+=delT
        self.steps+=1
        return should_continue
        
    def run(self):
        self.initialize_positions()
        isconv=False
        while(not isconv):
            should_continue=self.KMC_step()
            isconv=self.run_conv_checker() and should_continue
            
            
if(__name__=="__main__"):
    graph=nx.DiGraph()
    graph.add_node(1,occ=0,max_occ=1)
    graph.add_node(2,occ=0,max_occ=1)
    graph.add_node(0,occ=0,max_occ=1)
    graph.add_edge(0, 1, rate=1e7)
    graph.add_edge(1,2, rate=2e7)
    graph.add_edge(2,1, rate=2e7)
    graph.add_edge(0, 2, rate=3e7)
    graph.add_edge(2,0, rate=2e7)
    kmc=KMC(graph,convergence_criteria={"maxsteps":50})
    kmc.run()
        
        
        
        
        
        



