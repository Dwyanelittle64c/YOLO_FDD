def parse_model(d, ch):  # model_dict, input_channels(3)
    # LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    #
    # layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['YOLOX_head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # for j, a in enumerate(args):
        #     try:
        #         args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        #     except:
        #         pass
        #
        # n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]


        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)