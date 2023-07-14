import ipywidgets as widgets
from MainFunctions import *
from MapRxns import *

def visoutput(analoguerxns):
    if analoguerxns.index.name or any(analoguerxns.index.names):
        analoguerxns.reset_index(inplace=True)
    a = widgets.IntSlider(min=0,max=len(analoguerxns)-1,description='Row Number')
    # b = widgets.IntSlider(description='b')
    # c = widgets.IntSlider(description='c')
    def f(a):
        reaxysID=analoguerxns.iloc[a].ReactionID
        print('Reaxys reaction '+str(reaxysID)+':')
        if 'rxnsmiles0' in analoguerxns.dtypes:
            rxnsmiles0=analoguerxns.iloc[a].rxnsmiles0
            display(drawReaction(rdChemReactions.ReactionFromSmarts(rxnsmiles0,useSmiles=True)))
        if 'balrxnsmiles' in analoguerxns.dtypes:
            balrxnsmiles=analoguerxns.iloc[a].balrxnsmiles
            msg=analoguerxns.iloc[a].msg
            print('Balancing algorithm output: '+msg)
            print('Balanced reaction:')
            if balrxnsmiles!='Error':
                display(drawReaction(rdChemReactions.ReactionFromSmarts(balrxnsmiles,useSmiles=True)))
                print(balrxnsmiles)
            else:
                print('Error')
        if 'mapped_rxn' in analoguerxns.dtypes:
            mappedrxn=analoguerxns.iloc[a].mapped_rxn
        else:
            try:
                mappedrxn=maprxn([balrxnsmiles])[0]['mapped_rxn']
            except Exception:
                mappedrxn='Error'
        print('Mapped reaction:')
        if mappedrxn!='Error':
            display(drawReaction(rdChemReactions.ReactionFromSmarts(mappedrxn,useSmiles=True)))
        else:
            print(mappedrxn)
        if 'msg1' in analoguerxns.dtypes:
            msg1=analoguerxns.iloc[a].msg1
            print('Mapping validity: '+msg1)
        if 'template' in analoguerxns.dtypes:
            template=analoguerxns.iloc[a].template
            msg4=analoguerxns.iloc[a].msg4
            print('Template message: '+msg4)
            print('Template:')
            display(drawReaction(rdChemReactions.ReactionFromSmarts(template,useSmiles=True)))
            print('Template SMARTS: '+template)
    out = widgets.interactive_output(f, {'a': a})
    display(a)
    display(out)

def visoutput2(analoguerxns):
    #     breakpoint()
    if analoguerxns.index.name or any(analoguerxns.index.names):
        analoguerxns.reset_index(inplace=True)
    b = widgets.IntSlider(min=min(analoguerxns.ReactionID),max=max(analoguerxns.ReactionID),description='Reaxys ID')
    c = widgets.IntSlider(min=0,max=len(analoguerxns.loc[analoguerxns.ReactionID==b]),description='Instance')
    def f(b,c):
        reaxysID=b
        inst=c
        print('Reaxys reaction '+str(reaxysID)+':')
        if 'rxnsmiles0' in analoguerxns.dtypes:
            rxnsmiles0=analoguerxns.loc[analoguerxns.ReactionID==b].iloc[c].rxnsmiles0
            display(drawReaction(rdChemReactions.ReactionFromSmarts(rxnsmiles0,useSmiles=True)))
        if 'balrxnsmiles' in analoguerxns.dtypes:
            balrxnsmiles=analoguerxns.loc[analoguerxns.ReactionID==b].iloc[c].balrxnsmiles
            msg=analoguerxns.loc[analoguerxns.ReactionID==b].iloc[c].msg
            print('Balancing algorithm output: '+msg)
            print('Balanced reaction:')
            if balrxnsmiles!='Error':
                display(drawReaction(rdChemReactions.ReactionFromSmarts(balrxnsmiles,useSmiles=True)))
                print(balrxnsmiles)
            else:
                print('Error')
        if 'mapped_rxn' in analoguerxns.dtypes:
            mappedrxn=analoguerxns.loc[analoguerxns.ReactionID==b].iloc[c].mapped_rxn
        else:
            try:
                mappedrxn=maprxn([balrxnsmiles])[0]['mapped_rxn']
            except Exception:
                mappedrxn='Error'
        print('Mapped reaction:')
        if mappedrxn!='Error':
            display(drawReaction(rdChemReactions.ReactionFromSmarts(mappedrxn,useSmiles=True)))
        else:
            print(mappedrxn)
        if 'msg1' in analoguerxns.dtypes:
            msg1=analoguerxns.loc[analoguerxns.ReactionID==b].iloc[c].msg1
            print('Mapping validity: '+msg1)
        if 'template' in analoguerxns.dtypes:
            template=analoguerxns.loc[analoguerxns.ReactionID==b].iloc[c].template
            msg4=analoguerxns.loc[analoguerxns.ReactionID==b].iloc[c].msg4
            print('Template message: '+msg4)
            print('Template:')
            display(drawReaction(rdChemReactions.ReactionFromSmarts(template,useSmiles=True)))
    out = widgets.interactive_output(f, {'b': b,'c':c})
    display(b)
    display(c)
    display(out)