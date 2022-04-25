import numpy as np
import time
import os
from copy import deepcopy


# 3**12 * 160**12 * 18**4 * 30 * 35 * 140 * 16 > 10**30

def re_evaluations(x1):
    x = deepcopy(x1)

    M = 2
    C = 6

    for i in range(12):
        if x[i] > 0.66:
            x[i] = 25.0
        elif x[i] > 0.33:
            x[i] = 20.0
        else:
            x[i] = 15.0

    x[12:24] = np.round(list(40 + np.asarray(x[12:24]) * (200 - 40)))

    # C_1 to C_4 : one of the following values:
    C_values = [4.5, 4.7, 5, 5.6, 6.6, 6.8, 7.5, 8.2, 9.4, 10.0, 12.0, 14.0, 15.0, 17.0, 18.0, 19.0, 20.0, 22.0]

    for i in range(24, 28):
        t = x[i] * (22.0 - 4.5) + 4.5
        for j in range(1, len(C_values)):
            if C_values[j-1] == t:
                x[i] = C_values[j - 1]
            if C_values[j] == t:
                x[i] = C_values[j]
            if C_values[j - 1] < t < C_values[j]:
                if t - C_values[j - 1] > C_values[j] - t:
                    x[i] = C_values[j]
                else:
                    x[i] = C_values[j - 1]

    # l is one of the following:
    l_values = [.220, .240, .250, .270, .300, .330, .360, .390, .400, .420, .470, .500, .530, .560, .590, .600, .620,
                .680, .700, .760, .770, .820, .900, 1, 1.2, 1.5, 1.8, 2.2, 3.3, 4.7]

    t = x[28] * (4.7 - .220) + .220
    for j in range(1, len(l_values)):
        if l_values[j-1] == t:
            x[28] = l_values[j - 1]
        if l_values[j] == t:
            x[28] = l_values[j]
        if l_values[j - 1] < t < l_values[j]:
            if t - l_values[j - 1] > l_values[j] - t:
                x[28] = l_values[j]
            else:
                x[28] = l_values[j - 1]

    # cpo
    x[29] = np.round(200 + np.asarray(x[29]) * (6000 - 200)) // 100.0 * 100.0
    # ramp
    x[30] = np.round(200 + np.asarray(x[30]) * (3000 - 200)) // 20.0 * 20.0
    # vref
    x[31] = np.round(550 + np.asarray(x[31]) * (630 - 550)) // 5.0 * 5.0
    # # IL
    # x[32] = np.round(10 + np.asarray(x[32]) * (1200 - 10)) // 10 * 10

    print("f values (1-12): ", x[:12])
    print("m values (1-12): ", x[12:24])
    print("C values (1-4): ", x[24:28])
    print("l, cpo, ramp, vref:", x[28:])

    re_write_test_file(x)

    print("Starting the simulation")
    start_time = time.time()
    os.system('sh run_code')
    exists = False
    while not exists:
        print("wait for results file")
        time.sleep(2)
        exists = os.path.exists('hcrtestresult.txt')

    simulation_time = time.time() - start_time
    print("simulation_time", simulation_time)

    filename = "hcrtestresult.txt"
    f = open(filename, "r")
    f1 = f.readlines()
    Full_string = f1[2]

    parsing_line = Full_string.split()
    functions_values = []

    # print("output:", parsing_line)
    # ocnPrint(?output port eff Vo ripple vc1 vc2 vc3 vc4 vc5 vc6 vc7 Imin)

    paths = '.'

    with open(os.path.join(paths, 'results/Simulator2_output.txt'), "a") as filehandle:
        filehandle.write(' , '.join(parsing_line))
        filehandle.write('\n')
    filehandle.close()

    for i in range(len(parsing_line)):
        if 'm' in parsing_line[i]:
            functions_values.append(float(parsing_line[i][:-1]) * 1e-3)
        elif 'u' in parsing_line[i]:
            functions_values.append(float(parsing_line[i][:-1]) * 1e-6)
        elif 'n' in parsing_line[i]:
            functions_values.append(float(parsing_line[i][:-1]) * 1e-9)
        elif 'p' in parsing_line[i]:
            functions_values.append(float(parsing_line[i][:-1]) * 1e-12)
        elif 'k' in parsing_line[i]:
            functions_values.append(float(parsing_line[i][:-1]) * 1e3)
        elif 'K' in parsing_line[i]:
            functions_values.append(float(parsing_line[i][:-1]) * 1e3)
        else:
            functions_values.append(float(parsing_line[i]))

    f.close()
    os.remove("hcrtestresult.txt")
    print("number of returned values: ", len(functions_values))

    efficiency, V_out, ripple, Imin = functions_values[0], functions_values[1], functions_values[2], functions_values[
        -1]

    FoM = (x[24] + x[25] + x[26] + x[27]) / (V_out * 1000.0)

    transient_settling_time = 3.0 * 2000 * 3000 * 1e-9  # high value when infeasible

    vc_ea_cnt = (len(functions_values) - 4) // 2
    vcs = functions_values[3:3 + vc_ea_cnt]
    eas = functions_values[3 + vc_ea_cnt:-1]

    print("Efficiency: ", efficiency)
    print("Output voltage: ", V_out)
    print("ripple: ", ripple)
    print("minimum inductor current: ", Imin)

    if ripple >= 0.3 or V_out <= 0.3 or not 0 < efficiency <= 100:  # or functions_values[-1] <= 0:
        stability = -1
    else:
        stability = re_check_stable(eas, vcs, V_out)

        if stability < 0 and eas[519] <= eas[518] <= eas[517] <= eas[516] <= eas[515] <= eas[514] <= eas[513] <= eas[
            512] <= eas[511] <= eas[510] <= eas[509] or eas[519] >= eas[518] >= eas[517] >= eas[516] >= eas[515] >= eas[
            514] >= eas[513] >= eas[512] >= eas[511] >= eas[510] >= eas[509]:
            if ripple < 0.3 and V_out > 0.3 and 100.0 > efficiency >= 70.0:
                if abs(eas[519] - eas[509]) <= abs(eas[419] - eas[409]) <= abs(eas[319] - eas[309]) <= abs(
                        eas[219] - eas[209]):
                    stability = 520

    if stability >= 0:
        transient_settling_time = 3.0 * (stability + 1) * x[30] * 1e-9
        stability = 1

    V_ref = x[31]

    vo_verf_pos_cond = 10 - (V_out * 1000.0 - V_ref)
    vo_vref_neg_cond = 50 - (V_ref - V_out * 1000.0)
    eff_const = efficiency - 70
    ripple_const = 0.1 - ripple

    objectives = [efficiency, -transient_settling_time]
    constraints = [Imin, eff_const, ripple_const, vo_verf_pos_cond, vo_vref_neg_cond, stability]
    print("Vo: ", V_out, "Vref", V_ref)

    print("Efficiency: (0-100), -Transient settling time: ", objectives)  # minimizing all of these
    print("Imin, Efficiency - 70 , 0.1-ripple, 10 - (output voltage - reference voltage), 50 - (reference voltage - output voltage) , stability : ",
        constraints)  # >=0 and >0

    return x, objectives, constraints


def re_write_test_file(x):
    f = x[:12]
    m = x[12:24]
    C = x[24:28]
    l = x[28]
    cpo = x[29]
    ramp = x[30]
    vref = x[31]
    replacer = str(np.round(0.0004 + 1560 * (ramp * 1e-9), 15))
    file_in = "hcr_test.ocn"
    file_out = "tmp.ocn"
    vc_changed_line_flag = True
    ea_changed_line_flag = True
    with open(file_in, "rt") as fin:
        with open(file_out, "wt") as fout:
            for line in fin:
                if line.startswith("ocnPrint"):
                    vcs_str = ' '.join("vc" + str(i) for i in range(1, 521))
                    eas_str = ' '.join("ea" + str(i) for i in range(1, 521))
                    fout.write("ocnPrint(?output port eff Vo ripple " + vcs_str + " " + eas_str + " Imin)\n")
                elif line.startswith("ea"):
                    if ea_changed_line_flag:
                        ea_changed_line_flag = False
                        for cnt in range(520):
                            fout.write("ea" + str(cnt + 1) + " = average(clipX(vtime('tran \"/ea\") (0.0004 + (" + str(
                                3 * cnt) + " * VAR(\"ramp\"))) (0.0004 + (" + str(
                                3 * (cnt + 1)) + " * VAR(\"ramp\")))))\n")
                elif line.startswith("vc"):
                    if vc_changed_line_flag:
                        vc_changed_line_flag = False
                        for cnt in range(520):
                            fout.write("vc" + str(cnt + 1) + " = average(clipX(vtime('tran \"/vo\") (0.0004 + (" + str(
                                3 * (cnt)) + " * VAR(\"ramp\"))) (0.0004 + (" + str(
                                3 * (cnt + 1)) + " * VAR(\"ramp\")))))\n")
                elif line.startswith("analysis('tran"):
                    fout.write("analysis(\'tran ?stop \"" + replacer + "\"  ?cmin \"100f\"  )\n")
                elif line.startswith("ripple"):
                    fout.write(
                        "ripple = peakToPeak(clipX(vtime('tran \"/vo\") (" + replacer + " - (30 * VAR(\"ramp\"))) " + replacer + "))\n")
                elif line.startswith("Vo"):
                    fout.write(
                        "Vo = average(clipX(vtime('tran \"/vo\") (" + replacer + " - (30 * VAR(\"ramp\"))) " + replacer + "))\n")
                elif line.startswith("Iin"):
                    fout.write(
                        "Iin = average(clipX(IT(\"/V16/MINUS\") (" + replacer + " - (30 * VAR(\"ramp\"))) " + replacer + "))\n")
                elif line.startswith("Id"):
                    fout.write(
                        "Id = average(clipX(IT(\"/V14/MINUS\") (" + replacer + " - (30 * VAR(\"ramp\"))) " + replacer + "))\n")
                elif line.startswith("Imin"):
                    fout.write(
                        "Imin = ymin(clipX(IT(\"/V16/MINUS\") (" + replacer + " - (30 * VAR(\"ramp\"))) " + replacer + "))\n")
                elif line.startswith("desVar"):
                    splitted_line = line.split('"')
                    cur_var = splitted_line[1]
                    temp = splitted_line[2]
                    if cur_var == 'c1':
                        temp2 = temp.replace(temp[1:-3], str(C[0]) + "u")
                    elif cur_var == 'c2':
                        temp2 = temp.replace(temp[1:-3], str(C[1]) + "u")
                    elif cur_var == 'c3':
                        temp2 = temp.replace(temp[1:-3], str(C[2]) + "u")
                    elif cur_var == 'c4':
                        temp2 = temp.replace(temp[1:-3], str(C[3]) + "u")
                    elif cur_var == 'l':
                        temp2 = temp.replace(temp[1:-3], str(l) + "u")
                    elif cur_var == 'f1':
                        temp2 = temp.replace(temp[1:-3], str(int(f[0])))
                    elif cur_var == 'f2':
                        temp2 = temp.replace(temp[1:-3], str(int(f[1])))
                    elif cur_var == 'f3':
                        temp2 = temp.replace(temp[1:-3], str(int(f[2])))
                    elif cur_var == 'f4':
                        temp2 = temp.replace(temp[1:-3], str(int(f[3])))
                    elif cur_var == 'f5':
                        temp2 = temp.replace(temp[1:-3], str(int(f[4])))
                    elif cur_var == 'f6':
                        temp2 = temp.replace(temp[1:-3], str(int(f[5])))
                    elif cur_var == 'f7':
                        temp2 = temp.replace(temp[1:-3], str(int(f[6])))
                    elif cur_var == 'f8':
                        temp2 = temp.replace(temp[1:-3], str(int(f[7])))
                    elif cur_var == 'f9':
                        temp2 = temp.replace(temp[1:-3], str(int(f[8])))
                    elif cur_var == 'f10':
                        temp2 = temp.replace(temp[1:-3], str(int(f[9])))
                    elif cur_var == 'f11':
                        temp2 = temp.replace(temp[1:-3], str(int(f[10])))
                    elif cur_var == 'f12':
                        temp2 = temp.replace(temp[1:-3], str(int(f[11])))
                    elif cur_var == 'm1':
                        temp2 = temp.replace(temp[1:-3], str(int(m[0])))
                    elif cur_var == 'm2':
                        temp2 = temp.replace(temp[1:-3], str(int(m[1])))
                    elif cur_var == 'm3':
                        temp2 = temp.replace(temp[1:-3], str(int(m[2])))
                    elif cur_var == 'm4':
                        temp2 = temp.replace(temp[1:-3], str(int(m[3])))
                    elif cur_var == 'm5':
                        temp2 = temp.replace(temp[1:-3], str(int(m[4])))
                    elif cur_var == 'm6':
                        temp2 = temp.replace(temp[1:-3], str(int(m[5])))
                    elif cur_var == 'm7':
                        temp2 = temp.replace(temp[1:-3], str(int(m[6])))
                    elif cur_var == 'm8':
                        temp2 = temp.replace(temp[1:-3], str(int(m[7])))
                    elif cur_var == 'm9':
                        temp2 = temp.replace(temp[1:-3], str(int(m[8])))
                    elif cur_var == 'm10':
                        temp2 = temp.replace(temp[1:-3], str(int(m[9])))
                    elif cur_var == 'm11':
                        temp2 = temp.replace(temp[1:-3], str(int(m[10])))
                    elif cur_var == 'm12':
                        temp2 = temp.replace(temp[1:-3], str(int(m[11])))
                    elif cur_var == 'ramp':
                        temp2 = temp.replace(temp[1:-3], str(ramp) + "n")
                    elif cur_var == 'cpo':
                        temp2 = temp.replace(temp[1:-3], str(cpo) + "p")
                    elif cur_var == 'vref':
                        temp2 = temp.replace(temp[1:-3], str(vref) + "m")
                    else:
                        temp2 = temp
                    fout.write(line.split('"')[0] + '"' + line.split('"')[1] + '"' + temp2)
                else:
                    fout.write(line)

    os.remove(file_in)
    os.rename(file_out, file_in)


def re_check_stable(eas, vcs, vo):
    j = len(eas) - 1
    while j > 0 and eas[-1] - 0.004 <= eas[j] <= eas[-1] + 0.004:
        j -= 1
    if j >= len(eas) - 9:
        return -1
    i = len(vcs) - 1
    while i > j and (vo - 0.005 <= vcs[i] <= vo + 0.005):
        i -= 1
    if i >= len(eas) - 9:
        return -1
    return max(i, j)
