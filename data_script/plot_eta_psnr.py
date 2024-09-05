import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4.8))


x = [468]
y = [32.66]
# l = ax.plot(x, y, '.', label='FSRCNN', ms=10, color='tab:blue')

e = [0.5,   0.7,   0.75,  0.8,  0.805, 0.815, 0.82,  0.825, 0.835, 1.5,   2.7,   3.45,  3.95,  4.0]
x = [89,    119,   145,   174,  182,   187,   190,   195,   196,   224,   225,   249,   286,   290]
y = [32.38, 32.45, 32.53, 32.6, 32.64, 32.66, 32.68, 32.69, 32.7,  32.71, 32.72, 32.73, 32.74, 32.75]
# l = ax.plot(x, y, '-', label='ARM-FSRCNN', linewidth=2, color='tab:blue')

e = [0.0, 0.07894736842105263, 0.15789473684210525, 0.23684210526315788, 0.3157894736842105, 0.39473684210526316, 0.47368421052631576, 0.5526315789473684, 0.631578947368421, 0.7105263157894737, 0.7894736842105263, 0.8684210526315789, 0.9473684210526315, 1.026315789473684, 1.1052631578947367, 1.1842105263157894, 1.263157894736842, 1.3421052631578947, 1.4210526315789473, 1.5, 1.625, 1.75, 1.875, 2.0]
x = np.log([324, 304, 250, 231, 223, 216, 199, 175, 155, 140, 127, 116, 107, 101, 100, 100, 99, 98, 98, 97,  95, 92, 88, 82])
y = [32.7643, 32.7670, 32.7715, 32.7720, 32.7718, 32.7698, 32.7526, 32.7182, 32.6857, 32.6612, 32.6433, 32.6321, 32.6253, 32.6216, 32.6208, 32.6205, 32.6199, 32.6186, 32.6171, 32.6144, 32.6041, 32.5982, 32.5821, 32.5566]
l = ax.plot(e, y, '-', label='ENAF-FSRCNN', linewidth=2, color='tab:blue')

x = [1177]
y = [33.18]
# l = ax.plot(x, y, '.', label='CARN', ms=10, color='tab:orange')

e = [0.2,   0.5,   1.0,   1.08,  1.14,  1.16,  1.2,   1.24,  1.26,  1.3,   1.32, 1.34,  1.36,  1.38,  1.5,      1.94,  ]
x = [322,   353,   386,   421,   489,   504,  532,   601,   612,   683,      784,   ]
y = [33.05, 33.08, 33.09, 33.11, 33.18, 33.2, 33.22, 33.26, 33.27, 33.3,     33.31, ]
# l = ax.plot(x, y, '-', label='ARM-CARN', linewidth=2, color='tab:orange')

# Sampling patches rate:
e = [0.0, 0.15789473684210525, 0.3157894736842105, 0.47368421052631576, 0.631578947368421, 0.7894736842105263, 0.9473684210526315, 1.1052631578947367, 1.263157894736842, 1.4210526315789473, 1.5789473684210527, 1.7368421052631577, 1.894736842105263, 2.0526315789473684]
x = np.log([928, 834, 734, 663, 636, 623, 609, 598, 584, 574, 561, 539, 503, 451])
y = [33.3325, 33.3288, 33.3227, 33.3159, 33.3127, 33.3099, 33.3071, 33.3040, 33.3011, 33.2963, 33.2837, 33.2530, 33.1834, 33.0644]

l = ax.plot(e, y, '-', label='ENAF-CARN', linewidth=2, color='tab:orange')


x = [5324]
y = [33.50]
# l = ax.plot(x, y, '.', label='SRResNet', linewidth=3, ms=10, color='tab:red')

e = [0.2,   0.3,   0.5,   1.0,   1.4,   1.5,   1.6,  1.8,   1.9,   2.0,   4.0,   6.0,   7.5,   ]
x = [1218,  1220,  1243,  1295,  1321,  1352, 1391,  1520,  1832,  2406,  2869,  3243,  ]
y = [33.35, 33.36, 33.37, 33.39, 33.39, 33.4, 33.41, 33.43, 33.46, 33.49, 33.51, 33.52, ]
# l = ax.plot(x, y, '-', label='ARM-SRResNet', linewidth=2, color='tab:red')

e = [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714, 0.7142857142857143, 0.8571428571428571, 1.0, 1.1428571428571428, 1.2857142857142856, 1.4285714285714286, 1.5714285714285714, 1.7142857142857142, 1.8571428571428572, 2.0]
x = np.log([5059, 3399, 3262, 3121, 2789, 2384, 1941, 1563, 1431, 1421, 1419, 1418, 1418, 1418, 1418])
y = [33.5429, 33.5376, 33.5374, 33.5368, 33.5341, 33.5195, 33.4668, 33.3641, 33.2883, 33.2831, 33.2831, 33.2830, 33.2830, 33.2830, 33.2830]
l = ax.plot(e, y, '-', label='ENAF-SRResNet', linewidth=2, color='tab:red')


lgd = ax.legend()



ax.set_xlabel("$\eta$")
ax.set_ylabel('PSNR (db)')
ax.grid(alpha=0.5)
# labels = [item.get_text() for item in ax.get_yticklabels()]
# print(labels)
# labels = [0] + [f"$10^{i}$" for i in [5, 6, 7, 8]]
# ax.set_yticklabels(labels)

fig.savefig('eta_psnr.png', bbox_inches='tight', pad_inches=0.05)