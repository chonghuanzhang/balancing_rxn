# Possible types of error #

# LHS species insufficient, #LHS species not enough (reaction missing important reactants)
# RHS species insufficient, #Help products not sufficient to balance LHS (missing help products)
# Hydrogen carriers,      # Hydrogen deficit is present and hydrogen carriers are indicated but not added
# Charge imbalance,       # Charge imbalance present
# Mapping error,          # Mapping error (reaction SMILES string too long)
# discrepancy,            # SMILES discrepancy between mapped reactant and original supplied SMILES
# Mandatory               # Mandatory species on LHS unmapped
# Error                   #General error carried forward
# Invalid                 # Chempy has shifted reactants to RHS or vice versa thereby invalidating the reaction

#%%
from MainFunctions import *
# from helpCompound_full import *
from helpCompound import *
from BalanceRxns import *
from MapRxns import *
from visualisation import *

import pandas as pd

masterdbreadpath = '/home/aa2133/Impurity-Project/Reaxys_Data/'
inputparameters={'masterdbreadpath':'/home/aa2133/Impurity-Project/Reaxys_Data/', #Master reaxys data path
                 'substancesource':masterdbreadpath+'SubstanceSmiles.pickle', #Substance smiles path
                 'refbalrxns':None, #If prior results need to be included
                 'reagents':[], #Only if one reaction is inputted
                 'solvents':[], #Only if one reaction is inputted
                 'coefflim':6, #Maximum tolerable stoichiometric coefficient
                 'reaxys_update':True, #If Reaxys has been used to update
                 # 'reaxys_update': False, #If Reaxys has been used to update
                 'includesolv':True, #If solvents need to be used
                 'usemapper':True, #If mapper needs to be used
                 'helpprod':True, #If help products need to be used
                 'helpreact':False, #If help reactants need to be used
                 'addrctonly':False, # If only reactants should be included for balancing
                 'ignoreH':False, # If hydrogens are to be ignored when balancing
                 'ncpus':1, #Number of CPUs for computation
                 'restart':True, #Restart distributed cluster,
                 'helpprod':True, # Use help products or not
                 'helpreact':False, #Use help reactants or not
                 'hc_prod':hc_Dict, #Help compound dictionary,
                 'hc_react':None} #Help reactant dictionary

def masterbalance(rxns,IP=copy.deepcopy(inputparameters),**kwargs):
    if kwargs:
        for key,val in kwargs.items():
            if key in IP:
                IP[key]=val
    if type(rxns)==str: #Reaction SMILES inputted
        ncpus=1
        Rdata,Pdata,Rgtdata,Solvdata=getspecdat_rxn(rxns,reagents=IP['reagents'],solvents=IP['solvents'])
        rxns=pd.DataFrame([{'ReactionID':0,'Instance':0,'NumSteps':1,'NumStages':1,'NumRefs':1,'Rdata':Rdata,'Pdata':Pdata,
                            'Rgtdata':Rgtdata,'Solvdata':Solvdata,'hc_prod':IP['hc_prod'],'hc_react':IP['hc_react']}])
    else:
        if any([field not in rxns.dtypes for field in ['Rdata','Pdata','Rgtdata','Solvdata']]): #Database not yet updated
            rxns=addspeciesdata(rxns,IP['substancesource'],includesolv=IP['includesolv'],ncpus=IP['ncpus'],
                                restart=IP['restart'],reaxys_update=IP['reaxys_update'],hc_Dict=IP['hc_prod'],
                                hc_rct=IP['hc_react'])
            rxns['NumStages'] = 1
    balrxnsraw,balancedrxns=balance_analogue_(rxns,refbalrxns=IP['refbalrxns'],coefflim=IP['coefflim'],
                                              reaxys_update=IP['reaxys_update'],includesolv=IP['includesolv'],
                                              usemapper=IP['usemapper'],helpprod=IP['helpprod'],helpreact=IP['helpreact'],
                                              addrctonly=IP['addrctonly'],ignoreH=IP['ignoreH'],ncpus=IP['ncpus'],restart=IP['restart'])
    #     breakpoint()
    mappedrxns=map_rxns(balancedrxns,ncpus=IP['ncpus'],restart=IP['restart'],reaxys_update=IP['reaxys_update'])
    checkedrxns=checkrxns(mappedrxns,reaxys_update=IP['reaxys_update'],ncpus=IP['ncpus'])
    #     breakpoint()
    changedrxns=checkedrxns.loc[(checkedrxns.msg1.str.contains('Unmapped')) | (checkedrxns.msg1.str.contains('unmapped'))]
    changedrxns=changedrxns.loc[~(changedrxns.msg1.str.contains('Error')) & ~(changedrxns.msg1.str.contains('Invalid')) & ~(changedrxns.msg1.str.contains('Mandatory',case=False,na=False)) & ~(changedrxns.msg1.str.contains('discrepancy',case=False,na=False))]
    if not changedrxns.empty:
        changedrxns=updaterxns(changedrxns,hc_prod=IP['hc_prod'],analoguerxns=rxns,ncpus=IP['ncpus'])
        checkedrxns.update(changedrxns[['mapped_rxn','confidence','balrxnsmiles','msg','LHS','RHS','hcrct','hcprod','LHSdata','RHSdata','msg1']])
    return checkedrxns

#%%
if __name__ == '__main__':

    # Reaction source
    reactiondir='/home/aa2133/Impurity-Project/Input/Case4.4/DataMining/analoguerxns.pickle'
    # Updated reaction source (with reactant, product, reagent, solvent dictionaries)
    updatedreactiondir='/home/aa2133/Impurity-Project/Input/Case4.4/DataProcessing/analoguerxns_updated.pickle'
    # Substance database source
    substancesource=masterdbreadpath+'SubstanceSmiles.pickle'

    reactions=pd.read_pickle(reactiondir)

    updatedreactions=pd.read_pickle(updatedreactiondir)

    updatedreactionsample=updatedreactions[:1000]
    # updatedreactionsample
    # updatedreactionsample.dtypes
    # updatedreactionsample[['Rdata','Pdata','Rgtdata','Solvdata','hc_prod']]

    import time
    start = time.time()
    # finaldf=masterbalance('O=C(O)CS.Cc1cc(OC(C)c2nnc(N=Cc3ccccc3)s2)ccc1Cl>>Cc1cc(OC(C)c2nnc(N3C(=O)CSC3c3ccccc3)s2)ccc1Cl',ncpus=1)
    # finaldf

    # finaldf=masterbalance(updatedreactionsample,ncpus=1)

    finaldf=masterbalance(updatedreactionsample,ncpus=16)

    numbalanced=len(finaldf.loc[finaldf.msg.str.contains('balanced',case=False,na=False)])
    print(str(numbalanced)+' reactions were balanced')

    finish = time.time()
    print('time spent {} seconds'.format(finish-start))

    # visoutput(finaldf)
    # visoutput2(finaldf)

    finaldf.loc[finaldf.msg.str.contains('Balanced with help', case=False, na=False)]
