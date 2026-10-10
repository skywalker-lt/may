"""Minimum dense-equivalent expert size T_d (T4 ms, expert run on all tiles) for the routed design to clear the T4 bar,
as a function of the expert's efficiency e (AP per ms relative to the front slope; e=1 is on the front).
Routed gain = e*g*slope*T_d; routed cost = s*eps*eta*T_d + c. Clear if gain - slope*cost >= 0.003 + shortfall."""
slope=0.0102; s=0.16; eps=1.2; c=0.12
for g,lab in ((0.82,'g=0.82 (L, AP-level)'),(0.88,'g=0.88 (X, AP-level)')):
  for eta,hl in ((1.0,'halo-free'),(2.25,'conv halo 2.25')):
    for short,sl in ((0.0,'vs public front, no recipe loss'),(0.004,'with 0.004 recipe shortfall')):
        row=[]
        for e in (1.0,0.8,0.6,0.5,0.4,0.3):
            den=slope*(e*g-s*eps*eta)
            if den<=0: row.append(f'e={e}: never'); continue
            Td=(0.003+short+slope*c)/den; row.append(f'e={e}: Td>={Td:.2f} (routed +{s*eps*eta*Td+c:.2f} ms)')
        print(f'{lab}, {hl}, {sl}; threshold e>{s*eps*eta/g:.2f}:\n   '+' | '.join(row))
