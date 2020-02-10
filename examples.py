import robusta as rst

def mtcars():

    mtcars = rst.datasets.load('mtcars')
    mtcars['model'] = range(len(mtcars))
    anova = rst.Anova(
        data=mtcars, between=['gear', 'vs'],
        subject='model', dependent='qsec')


    return anova

