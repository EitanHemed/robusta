import robusta as rst


if __name__ == "__main__":
    ANXIETY_DATASET = rst.load_dataset('anxiety').set_index(
        ['id', 'group']).filter(
        regex='^t[1-3]$').stack().reset_index().rename(
        columns={0: 'score',
                 'level_2': 'time'})
    m = rst.groupwise.Anova(data=ANXIETY_DATASET, within='time',
                                between='group', dependent='score', subject='id')

    m.get_margins(margins_terms=['group'], by_terms=['time'])