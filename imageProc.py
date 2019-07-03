#%%
import numpy as np
# ------------------------------------------------------------------------------------------------ #
def nm2rgb(np_nm):
    ## CONSTANTS ##
    BNDS_NM = [380, 440, 490, 510, 580, 645, 780]
    dBNDS_NM = np.diff(BNDS_NM, 1)

    BNDS_INT = [340, 420, 700, 780]
    dBNDS_INT = np.diff(BNDS_INT, 1)

    SWATCH = [  [1/dBNDS_NM[0], 0, np.inf],
                [0, 1/dBNDS_NM[1], np.inf],
                [0, np.inf, 1/dBNDS_NM[2]],
                [1/dBNDS_NM[3], np.inf, 0],
                [np.inf, 1/dBNDS_NM[4], 0],
                [np.inf, 0, 0]                  ]
    COL_VAL = [ BNDS_NM[1], BNDS_NM[1], BNDS_NM[3], BNDS_NM[3], BNDS_NM[5], BNDS_NM[5] ]
    POLARITY = [-1, +1, -1, +1, -1, +1]

    ## Initialization ##
    rgb = np.zeros((len(np_nm), 3))
    Int = np.ones((len(np_nm), 1))

    # Assign each color #
    for l in range(len(np_nm)):
        # RGB Values #
        bin_nm = np.logical_and(np_nm[l] < BNDS_NM[1:], np_nm[l] >= BNDS_NM[:-1])
        for b in range(0, len(BNDS_NM)):
            if(bin_nm[b]):
                for c in range(3):
                    val = POLARITY[b] * (np_nm[l] - COL_VAL[b]) * SWATCH[b][c]
                    if(np.isnan(val)):
                        rgb[l,c] = 1
                    else:
                        rgb[l,c] = min(max(val, 0), 1)
                break
        
        # Intensity #
        if(BNDS_INT[0] <= np_nm[l] and np_nm[l] < BNDS_INT[1]):
            Int[l] = +(np_nm[l] - BNDS_INT[0])/dBNDS_INT[0]
        elif(BNDS_INT[2] <= np_nm[l] and np_nm[l] < BNDS_INT[3]):
            Int[l] = -(np_nm[l] - BNDS_INT[3])/dBNDS_INT[2]

    # Adjust values #
    rgb = Adjust(rgb, Int)
    return rgb
def rgb2nm(np_rgb):
    ## CONSTANTS ##
    INT_MAX = 255.0
    INT_TOL = 0.01  # Fraction of INT_MAX #

    ## Initialize ##
    nm = np.zeros(int(np.size(np_rgb)/3))
    rgb = np_rgb / INT_MAX

    if(len(nm) == 1):
        rgb_max = np.amax(rgb)
    else:
        rgb_max = np.amax(rgb, axis = 1)

    ## Find Wavelengths ##
    if(len(nm) == 1):
        # RED #
        if(rgb[0] > (1-INT_TOL)):                   # 780 to 580 #
            if(rgb[1] < (1-INT_TOL)):                   # 645 to 580 #
                nm = (1-rgb[1])*(645-580) + 580
            else:
                nm = 580
        # GREEN #
        elif(rgb[1] > (1-INT_TOL)):             # 580 to 490 #
            if(rgb[0] > INT_TOL):                       # 580 to 510 #
                nm = (0+rgb[0])*(580-510) + 510
            elif(rgb[2] > INT_TOL):                 # 510 to 490 #
                nm = (1-rgb[2])*(510-490) + 490
            else:
                nm = 510
        # BLUE #
        elif(rgb[2] > rgb_max*(1-INT_TOL)): # 490 to 380 #
            if(rgb[1] > INT_TOL):                       # 490 to 440 #
                nm = (0+rgb[1])*(490-440) + 440
            elif(rgb[0] > INT_TOL):                 # 440 to 380 #
                nm = rgb_max*(1-rgb[0])*(440-380) + 380
            else:
                nm = 440
    else:   
        for c in range(len(nm)):
            # RED #
            if(rgb[c,0] > (1-INT_TOL)):                 # 780 to 580 #
                if(rgb[c,1] < (1-INT_TOL)):                 # 645 to 580 #
                    nm[c] = (1-rgb[c,1])*(645-580) + 580
                else:
                    nm[c] = 580
            # GREEN #
            elif(rgb[c,1] > (1-INT_TOL)):               # 580 to 490 #
                if(rgb[c,0] > INT_TOL):                     # 580 to 510 #
                    nm[c] = (0+rgb[c,0])*(580-510) + 510
                elif(rgb[c,2] > INT_TOL):                   # 510 to 490 #
                    nm[c] = (1-rgb[c,2])*(510-490) + 490
                else:
                    nm[c] = 510
            # BLUE #
            elif(rgb[c,2] > rgb_max[c]*(1-INT_TOL)):    # 490 to 380 #
                if(rgb[c,1] > INT_TOL):                     # 490 to 440 #
                    nm[c] = (0+rgb[c,1])*(490-440) + 440
                elif(rgb[c,0] > INT_TOL):                   # 440 to 380 #
                    nm[c] = rgb_max[c]*(1-rgb[c,0])*(440-380) + 380
                else:
                    nm[c] = 440

    return nm
# ------------------------------------------------------------------------------------------------ #
def rgb2wsl(rgb):
    ## CONSTANTS ##
    INT_MAX = 255
    ERR_TOL = 0.00001

    print(rgb)

    # Extrast WSL #
    w = min(rgb)
    s = max(abs(rgb - w))
    l = rgb2nm( (rgb - w)/s * INT_MAX)

    print([w, s, l])
    print(nm2rgb(np.array([l]))[0])

    # Evaluate error #
    err = rgb - wsl2rgb( np.array([w,s,l]) )

    # Correct for negative errors #
    i = 1
    while((err < 0).any()):
        i = i+1
        derr = min(err[0]) / max(nm2rgb(np.array([l]))[0])

        s = s + i*derr
        err = rgb - wsl2rgb( np.array([w,s,l]) )

    ## KEEP ADDING NEW ONES UNTIL WE SATISFY THE TOLERANCE ##
    while(sum(abs(err[0])) > ERR_TOL):
        rgb_new = err[0]

        w_new = min(rgb_new)
        s_new = max(abs(rgb_new - w_new))
        l_new = rgb2nm( (rgb_new - w_new)/s_new * INT_MAX)

        w = np.append(w, w_new)
        s = np.append(s, s_new)
        l = np.append(l, l_new)

        fit = np.zeros((1,3))
        for c in range(len(w)):
            fit += wsl2rgb( np.array([w[c], s[c], l[c]]) )
            
        err = rgb - fit
        while((err < 0).any()):
            i = i+1
            derr = min(err[0]) / max(nm2rgb(np.array([l[-1]]))[0])

            s = s + i*derr
            fit = np.zeros((1,3))
            for c in range(len(w)):
                fit += wsl2rgb( np.array([w[c], s[c], l[c]]) )
            err = rgb - fit

    return np.array([w,s,l])
def wsl2rgb(wsl):
    ## CONSTANTS ##
    INT_MAX = 255

    # Compute the RGB value #
    rgb = (nm2rgb([wsl[2]]) / INT_MAX) * [wsl[1]] + [wsl[0]]
    return rgb
# ------------------------------------------------------------------------------------------------ #
def Adjust(color, intensity):
    ## CONSTANTS ##
    GAMMA = 0.80
    INT_MAX = 255

    ## Procedure ##
    return np.round(INT_MAX * np.power(color * intensity, GAMMA))
# ------------------------------------------------------------------------------------------------ #