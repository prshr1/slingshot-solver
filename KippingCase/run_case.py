from flyby import flyby_unpowered, flyby_oberth

G = 6.674e-11

# Example: choose physically meaningful units yourself (SI suggested)
xm0 = 12.0  # meters or scaled; use consistent units
ym0 = -1.0
um0 = 0.0
vm0 = -30000.0
vstar0 = 30000.0
M = 1.9885e30

out1 = flyby_unpowered(xm0, ym0, um0, vm0, vstar0, M, G)
print("NO OBERTH")
for k in ["vinf","b","e","rp","vp","theta","umF","vmF","dV"]:
    print(k, out1[k])

out2 = flyby_oberth(xm0, ym0, um0, vm0, vstar0, M, G, dv_peri=1000.0)
print("\nWITH OBERTH (dv_peri=1000 m/s)")
for k in ["vinf2","e2","rp","vp1","vp2","theta_tot","umF","vmF","dV"]:
    print(k, out2[k])
