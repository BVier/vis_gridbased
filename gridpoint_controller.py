from gridpoint_func import *

# TODO: Cell Layout einbauen?
# Decide if History plot or Single Point
appearance['hist'] = False



# Init Values
variables = {"s": '2', "mu": 'mu=4/log(i+1)', "i": 10}


plot()

# Subdomain controller
subax = fig.add_axes([0, 0.02, 0.1, 0.2])
subax.text(0.01,0.9, "Subdomains")
sub = RadioButtons(subax, subdomains, active=0)
sub.on_clicked(update_subdomains)

# Index controller
indexax = fig.add_axes([0.25, 0.02, 0.4, 0.03])
index = Slider(indexax, 'Repartition', 1, iterations, valinit=variables['i'], valstep=1)
index.on_changed(update_index)

#Mu controller
# - mu base
muax = fig.add_axes([0.8, 0.02, 0.1, 0.2])
muax.text(0,1.1, r'$\mu$')
muax.text(0.1,0.9, r'Base $c$')
mu_base = RadioButtons(muax, mu_base, active=1)
mu_base.on_clicked(update_mu_base)
# - mu divisor
divax = fig.add_axes([0.9, 0.02, 0.1, 0.2])
divax.text(0.1,0.9, "Divisor")
mu_div = RadioButtons(divax, ['/1', '/i', '/log(i+1)'], active=2)
mu_div.on_clicked(update_mu_div)


# Grid controller
checkax= fig.add_axes([0.25, 0.1, 0.25, 0.08])
grid_visible = CheckButtons(checkax, ['History view']) #'Grid visible',
grid_visible.on_clicked(update_grid)


plt.show()
