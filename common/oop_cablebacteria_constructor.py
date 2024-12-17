# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:00:57 2023

@author: smith
"""
#----------------------------------------ABOUT THIS FILE-------------------------------
#To "solve" a cable bacteria circuit from the starting point of #fibers #junctions Y/N bridged
#At the end we need to end up with a matrix for nodal analysis
#to construct a graph for nodal analysis we need a plausible representation of a circuit
#for this circuit representation in nodal analysis we will use a networkx graph as
#this is typical and convienent
#in this file we go from #fibers #junctions Y/N bridged to a series of connections representing a cable bacteria
#we first generate theis cable bacteria object instead of going straight to a graph as it makes the mapping easier
#this file also converts this cable bacteria object to a newtorkx graph
#---------------------------------------------------------------------------------------


import networkx as nx
import matplotlib.pyplot as plt


#----------------------------------CBNode----------------------------
#the most basic unit in the cable bacteria object
#this object stores some basic information about the node
#it also has the ability to be hashable allowing one to use the
#   CBNode as the key for a dictionary
class CBNode():
    def __init__(self,junction_id,fiber_id,nodeType="fiber"):
        """
        Cable Bacteria Node (CBNode) constructor

        Parameters
        ----------
        junction_id : INT
            Identifies the junction that this node is a part of
            For head/tail nodes this is expected to be -1
        fiber_id : INT/STR
            Identifies the junction that this node is a part of.
            For all fiber nodes this should be INT
            For a midpoint node in a junction this is expected to be "M"
            For head/tail nodes this is expected to be -1
        nodeType : STR, optional
            Should be either "fiber" (default) or "mid". This identifies
            if a given node is on a fiber or is teh midpoint of a junction

        Returns
        -------
        None.

        """
        self.junction_id=junction_id
        self.fiber_id=fiber_id
        self.nodeType=nodeType
        self.isMid=False
        self.isFib=False
        if(nodeType=="mid"):
            self.isMid=True
        if(nodeType=="fiber"):
            self.isFib=True
    
    #double underscore allowing this to be used as a key in a dictionary
    def __hash__(self):
        """
        Allows for CBNode to be a dictionary key

        Returns
        -------
        INT
            returns the hash of the string "f:fiber_id,j:junction_id"

        """
        return hash("f:"+str(self.fiber_id)+",j:"+str(self.junction_id))
#-----------------------------------------------------------------------
        

#----------------------------Junction-----------------------------------
#A junction object here is NOT the single node at the center of a junction
#Rather a junction object is the central node in a junction and all
#   of the directly connected nodes
#essentially it stores some identifying information and the node connections
#if the cable bacteria is not bridged it will not generate connections between
#   the central node and the periferal nodes
class Junction():
    def __init__(self,nfibers,uid,bridged=False):
        """
        Constructor for CB Junction object
        A junction object here is NOT the single node at the center of a junction
        Rather a junction object is the central node in a junction and all

        Parameters
        ----------
        nfibers : INT
            The number of fibers this CB has total
        uid : INT
            Unique identifier for this junction. Should be unqiue within a CB
        bridged : BOOL, optional
            If the CB is bridged or not. If false midpoint junctions will not be connected to fibers.
            The default is False.

        Returns
        -------
        None.

        """
        self.uid=str(uid)
        self.connections=[]
        self.midpoint=CBNode(uid,"M",nodeType="mid")
        #creates outer nodes around the inner bridging node (does not connect them here)
        self.outer_nodes=[CBNode(uid,str(i)) for i in range(nfibers)]
        self.bridged=bridged
        #generates connections if the CB is bridged
        if(self.bridged):
            self.connections=[[self.midpoint,self.outer_nodes[i]] for i in range(len(self.outer_nodes))]
        self.node_count=len(self.outer_nodes)+1
        
    #returns the CB node in this junction which is on the ith fiber
    #this is used later to properly generate all the connections between junctions
    def get_i_fiber(self,i):
        """
        returns the CB node in this junction which is on the ith fiber

        Parameters
        ----------
        i : INT
            The fiber number we are looking for a node on in this junction.

        Raises
        ------
        Exception
            If for some reason one is looking for a fiber that is not in this
            junction an exception is raised.

        Returns
        -------
        CBNode
            The CBNode which lies on fiber i.

        """
        for j in range(len(self.outer_nodes)):
            if(str(self.outer_nodes[j].fiber_id)==str(i)):
                return self.outer_nodes[j]
        raise Exception("Fiber not found in junction")
#-----------------------------------------------------------------------

#-------------------------CableBacteria---------------------------------  
#final representation and generation of a cable bacteria
#the __init__ generates the object representation for CB
#must call to_graph() to get networkx representation
class CableBacteria():
    
    def __init__(self,nfibers,mjunctions,bridged=True):
        """
        Constructor for CableBacteria object
        Also generates the CB in object form
            must call to_graph() to get the
            networkx representation

        Parameters
        ----------
        nfibers : INT
            Number of fibers in CB.
        mjunctions : TYPE
            Number of junctions in CB.
        bridged : TYPE, optional
            If the junctions should have connections between fibers.
            The default is True.

        Returns
        -------
        None.

        """
        #save inputs
        self.nfibers=nfibers
        self.mjunctions=mjunctions
        self.bridged=bridged
        
        #initialize book keeping
        self.junction_count=0
        self.junction_list=[]
        self.connections=[] #tracks all the connections between CBNodes in CB
        
        #add head and tail nodes
        self.head_node=CBNode(-1,-1,nodeType="head")
        self.tail_node=CBNode(-1,-1,nodeType="tail")
        
        #add junctions including attaching to head
        for i in range(mjunctions):
            self.addjunct()
        #attach last junction to tail
        self.close_cable()
        
    def addjunct(self):
        """
        Add a junction to the CB

        Returns
        -------
        None.

        """
        #initialize the new junction
        newjunction=Junction(self.nfibers,self.junction_count,bridged=self.bridged)
        #if this is the first junction attach to the head
        if(self.junction_count==0):
            for o in newjunction.outer_nodes:
                self.connections.append([self.head_node,o])
        else:
            #other wise generate connections between nodes on the same fibers on adjancent junctions
            for i in range(self.nfibers):
                #[-1] attaches to the previous junction
                self.connections.append([self.junction_list[-1].get_i_fiber(i),newjunction.get_i_fiber(i)])
        #append and iterate
        self.junction_count+=1
        self.junction_list.append(newjunction)
        
    def close_cable(self):
        """
        Attaches the current last junction to the tail node

        Returns
        -------
        None.

        """
        for i in range(self.nfibers):
            self.connections.append([self.junction_list[-1].get_i_fiber(i),self.tail_node])
    
    #Notes on graphing:
    #for visualizing CB graphs the following snippets may be useful
    #nx.draw(na.graph,with_labels = True,color_map=["green" if na.graph.degree(n)==cc.njunctions else 'blue' for n in na.graph.nodes])
    
    def to_graph(self,colorful=False,createReverseMap=False):
        """
        stores a networkx graph in self.graph that represents the CB
        before call self.graph does not exist (similarly creates self.toNumbMap)

        Parameters
        ----------
        colorful : BOOL, optional
            Adds color weights to the ntworkx graph to make later visualization easier
            wont work if need more than 4 colors. 
            NOTE This does NOT generate a networkx drawing
            To do so see above "Notes on graphing"
            The default is False.
            
        createReverseMap : BOOL, optional
            If True creates a dictionary stored in self.revMap.
            The keys of this dict are the networkx node labels
            The values are the specific CBNode object that networkx
                node corresponds to
            By default this is False as for large graphs this is a memory expensive object
            The default is False.

        Returns
        -------
        None. But creates and populates the variable self.graph (also creates self.toNumbMap)

        """
        if(colorful):
            #wont work if need more than 4 colors
            #add more colors here
            jcolors=["#BE2D1A","#3B7E00","#2C87FE","#FFB100"]
            fcolors=["#d9d2e9","#c3d0b3","#f48b7c","#93acff"]
            clists=[jcolors,fcolors]
            #---DEPRECIATED---
            self.gcolors=[]
            self.gcoords=[]
            #-----------------
            #-----COLOR-KEY----
            #generates a plot of labeled color points
            #the position of these points does not mater
            #rather the points in the plot are labeled 
            #with text saying what junction or fiber that color
            #represents
            #NOTE at NO point does this method create a networkx drawing
            #To do so see above "Notes on graphing"
            for i in range(len(clists)):
                for j in range(len(clists[i])):
                    ctyp="fib"
                    if(i==0):
                        ctyp="junc"
                    plt.scatter(i,j,color=clists[i][j])
                    plt.text(i,j,ctyp+str(j),color=clists[i][j])
            #-----------------
        #-------Object rep to networkx rep mapping-------
        #we need some mapping to consistently go between the two in a well
        #   defined way so we generate the dict self.toNumbMap
        self.toNumbMap={self.head_node:1,self.tail_node:"G"}#add head and tail nodes to mapping
        self.revMap={}
        if(createReverseMap):
            self.revMap[1]=self.head_node
            self.revMap["G"]=self.tail_node
        
        self.node_count=2 #head node add so start at 2
        #iterate over all nodes by going through the junctions
        #the only nodes that arent related (not strictly connected) to a junction
        #   are the head/tail which are handled above
        for junct in self.junction_list:
            #add fiber nodes
            for onode in junct.outer_nodes:
                self.toNumbMap[onode]=self.node_count
                if(createReverseMap):
                    self.revMap[self.node_count]=onode
                self.node_count+=1
            #add junction center node
            self.toNumbMap[junct.midpoint]=self.node_count
            if(createReverseMap):
                self.revMap[self.node_count]=onode
            self.node_count+=1
        self.node_count+=1 #add tail node
        
        self.graph=nx.Graph()#initialize blank graph
        #add fixed components
        #meaning components that are the same independent of the number of junctions
        #   number of fibers and if the CB is bridged
        #Generally this is the test resistor and the voltage
        self.graph.add_weighted_edges_from([('G', 0, 0)])
        self.graph.edges['G', 0]["typ"]="V"
        self.graph.edges['G', 0]["voltage"]=-1
        self.graph.edges['G', 0]["color"]="black"
        self.graph.edges['G', 0]["s"]="V"
        self.graph.add_weighted_edges_from([(0, 1, -1)])
        self.graph.edges[0, 1]["typ"]="R"
        self.graph.edges[0, 1]["s"]="RT"
        self.graph.edges[0, 1]["color"]="black" 
        
        #connect all the fiber nodes together in networkx
        for con in self.connections:
            self.graph.add_weighted_edges_from([(
                self.toNumbMap[con[0]],
                self.toNumbMap[con[1]],
                -1
                )])
            if(colorful):
                self.graph.edges[self.toNumbMap[con[0]], self.toNumbMap[con[1]]]["color"]=fcolors[int(con[0].fiber_id)]
        
        #connect all the junction center networkx nodes
        #to the surrounding fiber networkx nodes
        for junct in self.junction_list: 
            for con in junct.connections:
                self.graph.add_weighted_edges_from([(
                    self.toNumbMap[con[0]],
                    self.toNumbMap[con[1]],
                    -1
                    )])
                if(colorful):
                    self.graph.edges[self.toNumbMap[con[0]], self.toNumbMap[con[1]]]["color"]=jcolors[int(con[0].junction_id)]
    
