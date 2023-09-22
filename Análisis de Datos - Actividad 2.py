# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:34:50 2023

@author: jd.vasquezh
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

nombres_archivos =["Act 2(Al 0.1mm)", "Act 2(Al 0.08mm)", "Act 2(Al 0.06mm)",
                   "Actividad 2 (Al 0.04)", "Actividad 2 (Al 0.02mm)", "Actividad 2 (no material)","Act 2(Zn 0.1mm)","Act 2(Zn 0.075mm)",
                   "Act 2(Zn 0.05mm)","Act 2(Zn 0.025)", "Actividad 2 (no material)"]
intensidades = np.zeros((11,11))
k = 0
for nombre_archivo in nombres_archivos:
    archivo = open('./Datos/'+nombre_archivo, 'r')
    
    #Se omite la información de títulos y unidades de medida
    for i in range(3):
        archivo.readline()
        
    #Se extraen los datos al formato necesario
    angulo = []
    j = 0
    for linea in archivo.readlines():
        datos = linea.split("\t")
        angulo.append(float(datos[0].replace(",",".")))
        intensidades[k][j] = (float(datos[1][:-1].replace(",",".")))
        j += 1
    angulo = np.array(angulo)
    archivo.close()
    k += 1

espesoresAl = np.array([0.1, 0.08, 0.06, 0.04, 0.02, 0])
espesoresZn = np.array([0.1, 0.75, 0.05, 0.025, 0])

longitudesAl = []
longitudesZn = []
i = 0
for i in range(len(intensidades[0])):
    longitudesAl.append(intensidades[:6,i]/np.max(intensidades[:6,i]))
    longitudesZn.append(intensidades[7:,i]/np.max(intensidades[7:,i]))

def f(x, m):
    return np.exp(-x*m)

coeficientes = []
plt.figure()
for datos in longitudesAl:
    popt_reg, pcov_reg = curve_fit(f, espesoresAl, datos)
    perr_reg = np.sqrt(np.diag(pcov_reg))
    print("linear absorption coefficient = %0.2f (+/-) %0.2f" % (popt_reg[0], perr_reg[0]))
    coeficientes.append(popt_reg[0])
    plt.plot(espesoresAl,datos, "ro", markersize=3)
    plt.plot(espesoresAl,f(espesoresAl, popt_reg), 'k--')
plt.show()

longitud_onda = []
for t in angulo:
    longitud_onda.append(2*2.014*10**(-10)*np.sin(np.radians(t))**3)
plt.figure()
longitud_onda = np.array(longitud_onda)
coeficientes = np.array(coeficientes)
plt.plot(angulo[:-1], coeficientes[:-1], 'ro')
plt.show()
plt.plot(longitud_onda[:-1], coeficientes[:-1], 'ro')
plt.show()
'''
i = 1
while i < len(experimentos):
    if i >= 1 and i <= 5:
        while i < len
        longitudesAl.append(experimento[0] experimentos[i])

datos = experimentos[0]


'''
