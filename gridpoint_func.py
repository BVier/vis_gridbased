import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from math import log

# Init Values
variables = {"s": '2', "mu": 'mu=4/log(i+1)', "i": 10}
appearance = {'grid':False, 'hist':False}


'''
Static global values
'''

box_len = 80
iterations = 40
mu_base = ['1', '4', '10']
mu_div = ['', '/i', '/log(i+1)']
subdomains = ['2', '4', '8']

with open('40gp_relaxed_detail.json') as json_data:
    data = json.load(json_data)


# Figure to plot in
fig = plt.figure()
# 3d axis
ax = Axes3D(fig, [0.1,0.1,0.8, 0.9])


'''
Helper functions
'''

def get_data(value, v=variables):
    # value must be in ['Weight', 'Midpoints', 'Shift_vec', 'Gridpoints', 'Imbalance']
    # or any self defined label
    return np.array(data[v['s']][v['mu']][v['i']][value])


def mu_splitter(mu):
    split = mu.find('/')
    if split == -1:
        return mu, ''
    return mu[:split], mu[split:]


def mu_of(i):
    mu, div = mu_splitter(variables['mu'])
    base = int(mu[:3])
    factor = 1
    if div == '/log(i+1)':
        factor /= log(i+1)
    elif div == '/i':
        factor /= i
    return base*factor


def shape_for_subdomains(s):
    dims = np.ones((4,))
    subd = int(s)
    dims[0] = 5 if subd == 16 else 3
    dims[1] = 3 if subd % 4 == 0 else 2
    dims[2] = 3 if subd % 8 == 0 else 2
    dims[3] = 3
    return dims.astype(int)


def init_gridpoints(s):
    s = int(s)
    # shape = np.ones(4)
    # shape += np.array([s % 16 == 0, s % 4 == 0, s % 8 == 0, False]).astype(int)
    # shape *= np.array([2, 1, 1, 3])
    shape = shape_for_subdomains(s) - np.array([1,1,1,0])
    init_gps = np.ones(shape)*box_len
    for x in range(shape[0]):
        init_gps[x, :, :, 0] = init_gps[x, :, :, 0]*(x+1)/shape[0]
    for y in range(shape[1]):
        init_gps[:, y, :, 1] = init_gps[:, y, :, 1]*(y+1)/shape[1]
    for z in range(shape[2]):
        init_gps[:, :, z, 2] = init_gps[:, :, z, 2]*(z+1)/shape[2]
    return init_gps.reshape([s, 3])


def add_old_gridpoints(d=data):
    v = {}
    for s in subdomains:
        v["s"] = s
        for base in mu_base:
            for div in mu_div:
                mu = "mu=" + base + div
                v['mu'] = mu
                d[s][mu][0]['oldGridpoints'] = init_gridpoints(s)
                for i in range(iterations - 1):
                    v["i"] = i
                    d[s][mu][i+1]['oldGridpoints'] = get_data('Gridpoints', v)


def mirror(points, shift=box_len):
    s = variables['s']
    shape = shape_for_subdomains(s)
    vertices = np.zeros(shape)

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                pos = np.array([x, y, z])
                shift_dim = (pos == 0).astype(int)
                pos += shift_dim * (shape[:3] - 1)

                i = pos-1
                i %= 2
                i *= np.array([1, 2, 4])
                i = np.sum(i)
                order = {'2': [0, 1, 2], '4': [1, 0, 2],
                         '8': [2, 1, 0], '16': [0, 1, 2]}
                vertices[x, y, z] = points[i, order[s]] - shift_dim*shift

    return vertices


'''
Plot Functions
'''

def plot_plane(plane):
    # Sadly not used
    plane_line = []
    dims = plane.shape
    for y in range(dims[1] - 1):
        for x in range(dims[0]-1):
            plane_line.append(plane[x, y])
        for x in range(dims[0], 0, -1):
            plane_line.append(plane[x-1, y])
            plane_line.append(plane[x-1, y+1])
    for x in range(dims[0]):
        plane_line.append(plane[x, dims[1]-1])
    plane_line = np.array(plane_line)
    ax.plot(plane_line[:, 0], plane_line[:, 1],
            plane_line[:, 2], alpha=0.1, c='grey')


def plot_domains(vertices, alf = 0.4):
    dims = vertices.shape
    for x in range(dims[0]):
        for y in range(dims[1]):
            ax.plot(vertices[x, y, :, 0], vertices[x, y, :, 1],
                    vertices[x, y, :, 2], alpha=alf, c='grey')
    for y in range(dims[1]):
        for z in range(dims[2]):
            ax.plot(vertices[:, y, z, 0], vertices[:, y, z, 1],
                    vertices[:, y, z, 2], alpha=alf, c='grey')
    for z in range(dims[2]):
        for x in range(dims[0]):
            ax.plot(vertices[x, :, z, 0], vertices[x, :, z, 1],
                    vertices[x, :, z, 2], alpha=alf, c='grey')


def plot_shift_vec( color='green'):
    vertices = mirror(get_data('oldGridpoints'))
    shifted_vertices = vertices + mirror(get_data("Shift_vec"), 0)
    shifted_vertices = mirror(get_data('Gridpoints'))
    shape = vertices.shape
    number_of_vertices = shape[0]*shape[1]*shape[2]

    summary = np.array([vertices, shifted_vertices])
    summary.shape = [2, number_of_vertices, 3]
    summary[1] += (summary[1]-summary[0]) *20
    for vec in range(number_of_vertices):
        ax.plot(summary[:, vec, 0], summary[:, vec, 1],
                summary[:, vec, 2], c=color)


def plot_weight_points():
    midpoints = get_data('Midpoints')
    weight = get_data("Weight")
    imb = weight / np.mean(weight)
    ax.scatter(midpoints[:, 0], midpoints[:, 1],
               midpoints[:, 2], str(weight), c=imb, s=250*imb**2)
    return np.max(imb)


def plot():
    ax.clear()
    if not appearance['grid']:
        ax.set_axis_off()
    if appearance['hist']:
        for step in range(0,iterations-1, variables['i']):
            variables['i'] = step
            alpha = (step+1 )/iterations
            vertices = mirror(get_data('oldGridpoints'))
            plot_domains(vertices, alpha)
    else:
        vertices = mirror(get_data('oldGridpoints'))
        ax.scatter(vertices[:, :, :, 0],
                vertices[:, :, :, 1], vertices[:, :, :, 2])
        plot_domains(vertices)
        plot_shift_vec()
        print(get_data("Imbalance"))
        imb = plot_weight_points()
        plt.title("Imbalance = {}".format(imb))


'''
Control Functions
'''

def update_index(i):
    variables['i'] = int(i) -1
    plot()


def update_subdomains(subd):
    variables['s'] = subd
    plot()


def update_mu_base(mu):
    base, div = mu_splitter(variables['mu'])
    base = 'mu=' + mu
    variables['mu'] = base+div
    plot()


def update_mu_div(div):
    base, old_dv = mu_splitter(variables['mu'])
    div = '' if div == '/1' else div
    variables['mu'] = base+div
    plot()


def update_grid(vis):
    if(vis=='Grid visible'):
        appearance['grid'] = not appearance['grid']
    if(vis=='History view'):
        appearance['hist'] = not appearance['hist']
    plot()


add_old_gridpoints()
