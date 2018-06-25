from topology import SingleCell

if __name__ == '__main__':
    radius = 500  # m
    cue_num = 10
    d2d_num = 20
    rb_num = 10
    up_or_down_link = 'up'
    d_tx2rx = 10  # m
    single_cell = SingleCell(radius, cue_num, d2d_num, rb_num, up_or_down_link, d_tx2rx)
    single_cell.initial()
    single_cell.initial_channel()
    pass
