
mobile_opt = {"0": None,
              "small": [
                # t, c, n, s
                [1, 12, 1, 1],
                [6, 12, 2, 2],
                [6, 10, 3, 2],
                [6, 8, 4, 2],
                [6, 9, 3, 1],
                [6, 12, 3, 2],
                [6, 315, 1, 1],],
            "middle": [
                # t, c, n, s
                [1, 12, 1, 1],
                [6, 12, 2, 2],
                [6, 13, 3, 2],
                [6, 16, 4, 2],
                [6, 18, 3, 1],
                [6, 22, 3, 2],
                [6, 315, 1, 1],],
            "big": [
                # t, c, n, s
                [1, 12, 1, 1],
                [6, 12, 2, 2],
                [6, 16, 3, 2],
                [6, 20, 4, 2],
                [6, 24, 3, 1],
                [6, 36, 3, 2],
                [6, 316, 1, 1],],
            "huge": [
                # t, c, n, s
                [1, 12, 1, 1],
                [6, 18, 2, 2],
                [6, 24, 3, 2],
                [6, 42, 4, 2],
                [6, 60, 3, 1],
                [6, 96, 3, 2],
                [6, 318, 1, 1],],
            "enormous": [
                # t, c, n, s
                [1, 14, 1, 1],
                [6, 24, 2, 2],
                [6, 28, 3, 2],
                [6, 48, 4, 2],
                [6, 72, 3, 1],
                [6, 120, 3, 2],
                [6, 318, 1, 1],],
            "huge_smallt":
            [
                  # t, c, n, s
                  [1, 12, 1, 1],
                  [4, 18, 2, 2],
                  [4, 24, 3, 2],
                  [3, 42, 4, 2],
                  [3, 60, 3, 1],
                  [2, 96, 3, 2],
                  [2, 318, 1, 1]
              ],
            "huge_bigt":
            [
                  # t, c, n, s
                  [1, 12, 1, 1],
                  [5, 18, 2, 2],
                  [4, 24, 3, 2],
                  [4, 42, 4, 2],
                  [3, 60, 3, 1],
                  [3, 96, 3, 2],
                  [2, 318, 1, 1]]
}


seresnet_cfg = {"0": None,
                "cfg1": "config/pose_cfg/seresnet_cfg.txt",
                }


efficientnet_cfg = {str(i): "b"+str(i) for i in range(9)}

shufflenet_cfg = {0: ""}


DUC_cfg = {0: [640, 320],
           1: [480, 240],
           2: [320, 160],
           }

