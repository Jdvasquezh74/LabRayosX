import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

archivo = open('./Datos/Actividad 1', 'r')

#Se omite la información de títulos y unidades de medida
for i in range(3):
    archivo.readline()

#Se extraen los datos al formato necesario
angulo = []
intensidad = []
for linea in archivo.readlines():
    datos = linea.split("\t")
    angulo.append(float(datos[0].replace(",",".")))
    intensidad.append(float(datos[1][:-1].replace(",",".")))

archivo.close()

angulo = np.array(angulo)
intensidad = np.array(intensidad)

#Cálculo de un ajuste lorentziano para el modelo

#Se define la función para un único pico lorentziano para separar cada pico en la figura
def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)

#Se define la función para 4 picos con ajuste lorentziano
def _4Lorentzian(x, amp1,cen1,wid1, amp2,cen2,wid2, amp3,cen3,wid3, amp4,cen4,wid4):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
           (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
           (amp3*wid3**2/((x-cen3)**2+wid3**2)) +\
           (amp4*wid4**2/((x-cen4)**2+wid4**2))

#Se usa la gráfica de intensidad respecto al ángulo para analizar una aproximación 
#visual a los parámetros
amp1 = 1765
cen1 = 17.5
wid1 = 1

amp2 = 6636
cen2 = 19.8
wid2 = 1

amp3 = 411
cen3 = 41
wid3 = 1

amp4 = 1261
cen4 = 47.1
wid4 = 1

indices_lorentziano1 = np.where((angulo >= 17.0) & (angulo <= 18.0))
angulo_l1 = angulo[indices_lorentziano1]
intensidad_l1 = intensidad[indices_lorentziano1]

indices_lorentziano2 = np.where((angulo >= 19.0) & (angulo <= 20.5))
angulo_l2 = angulo[indices_lorentziano2]
intensidad_l2 = intensidad[indices_lorentziano2]

indices_lorentziano3 = np.where((angulo >= 40.5) & (angulo <= 41.4))
angulo_l3 = angulo[indices_lorentziano3]
intensidad_l3 = intensidad[indices_lorentziano3]

indices_lorentziano4 = np.where((angulo >= 46.5) & (angulo <= 47.7))
angulo_l4 = angulo[indices_lorentziano4]
intensidad_l4 = intensidad[indices_lorentziano4]

#Se calculan por separado los parámetros para el ajuste lorentziano de cada pico

#Cálculo de parámetros pico 1
popt_lorentz1, pcov_lorentz1 = curve_fit(_1Lorentzian, angulo_l1, intensidad_l1, p0=[amp1, cen1, wid1])

perr_1lorentz = np.sqrt(np.diag(pcov_lorentz1))

pars_1 = popt_lorentz1[0:3]
lorentz_peak_1 = _1Lorentzian(angulo, *pars_1)

#Cálculo de parámetros pico 2
popt_lorentz2, pcov_lorentz2 = curve_fit(_1Lorentzian, angulo_l2, intensidad_l2, p0=[amp2, cen2, wid2])

perr_2lorentz = np.sqrt(np.diag(pcov_lorentz2))

pars_2 = popt_lorentz2[0:3]
lorentz_peak_2 = _1Lorentzian(angulo, *pars_2)

#Cálculo de parámetros pico 3
popt_lorentz3, pcov_lorentz3 = curve_fit(_1Lorentzian, angulo_l3, intensidad_l3, p0=[amp3, cen3, wid3])

perr_3lorentz = np.sqrt(np.diag(pcov_lorentz3))

pars_3 = popt_lorentz3[0:3]
lorentz_peak_3 = _1Lorentzian(angulo, *pars_3)

#Cálculo de parámetros pico 4
popt_lorentz4, pcov_lorentz4 = curve_fit(_1Lorentzian, angulo_l4, intensidad_l4, p0=[amp4, cen4, wid4])

perr_4lorentz = np.sqrt(np.diag(pcov_lorentz4))

pars_4 = popt_lorentz4[0:3]
lorentz_peak_4 = _1Lorentzian(angulo, *pars_4)

#Parámetros del ajuste junto con los errores de cada uno
print("-------------Peak 1-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_1[0], perr_1lorentz[0]))
print("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_1lorentz[1]))
print("width = %0.2f (+/-) %0.2f" % (pars_1[2], perr_1lorentz[2]))
print("area = %0.2f" % np.trapz(lorentz_peak_1))
print("--------------------------------")
print("-------------Peak 2-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_2[0], perr_2lorentz[0]))
print("center = %0.2f (+/-) %0.2f" % (pars_2[1], perr_2lorentz[1]))
print("width = %0.2f (+/-) %0.2f" % (pars_2[2], perr_2lorentz[2]))
print("area = %0.2f" % np.trapz(lorentz_peak_2))
print("--------------------------------")
print("-------------Peak 3-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_3[0], perr_3lorentz[0]))
print("center = %0.2f (+/-) %0.2f" % (pars_3[1], perr_3lorentz[1]))
print("width = %0.2f (+/-) %0.2f" % (pars_3[2], perr_3lorentz[2]))
print("area = %0.2f" % np.trapz(lorentz_peak_3))
print("--------------------------------")
print("-------------Peak 4-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_4[0], perr_4lorentz[0]))
print("center = %0.2f (+/-) %0.2f" % (pars_4[1], perr_4lorentz[1]))
print("width = %0.2f (+/-) %0.2f" % (pars_4[2], perr_4lorentz[2]))
print("area = %0.2f" % np.trapz(lorentz_peak_4))
print("--------------------------------")

#Se hallan los residuales para el ajuste obtenido, tomando solo aquellos
#significativos para los picos
popt_lorentz = np.concatenate((popt_lorentz1,popt_lorentz2,popt_lorentz3,popt_lorentz4),axis=None)
residual_4lorentz = intensidad - (_4Lorentzian(angulo, *popt_lorentz))
indices_residuales = np.where((angulo < 10))
residual_4lorentz[indices_residuales] = np.nan

#Se obtiene la gráfica para los datos arrojados
fig = plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
gs.update(hspace=0) 

ax1.plot(angulo, intensidad, "ro", markersize=3)
ax1.plot(angulo, _4Lorentzian(angulo, *popt_lorentz), 'k--')#,\
         #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))
'''
# peak 1
ax1.plot(angulo, lorentz_peak_1, "g")
ax1.fill_between(angulo, lorentz_peak_1.min(), lorentz_peak_1, facecolor="green", alpha=0.5)
  
# peak 2
ax1.plot(angulo, lorentz_peak_2, "y")
ax1.fill_between(angulo, lorentz_peak_2.min(), lorentz_peak_2, facecolor="yellow", alpha=0.5)  

# peak 3
ax1.plot(angulo, lorentz_peak_3, "m")
ax1.fill_between(angulo, lorentz_peak_3.min(), lorentz_peak_3, facecolor="purple", alpha=0.5) 

# peak 4
ax1.plot(angulo, lorentz_peak_4, "c")
ax1.fill_between(angulo, lorentz_peak_4.min(), lorentz_peak_4, facecolor="cyan", alpha=0.5) 
'''

#Plot para residuales
ax2.plot(angulo, residual_4lorentz, "bo", markersize=3)
    
#ax1.set_xlim(-5,105)
#ax1.set_ylim(-0.5,8)

ax2.set_xlim(0,60)
#ax2.set_ylim(-0.5,0.75)

ax2.set_xlabel("Ángulo $\\theta$ (°)",family="serif",  fontsize=12)
ax1.set_ylabel("Intensidad - R 35kV - (Imp/s)",family="serif",  fontsize=12)
ax2.set_ylabel("Res.",family="serif",  fontsize=12)

ax1.legend(loc="best")

ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.xaxis.set_major_formatter(plt.NullFormatter())

ax1.tick_params(axis='x',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='x',which='minor', direction="out", top="on", right="on", bottom="off", length=5, labelsize=8)
ax1.tick_params(axis='y',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='y',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

ax2.tick_params(axis='x',which='major', direction="out", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='x',which='minor', direction="out", top="off", right="on", bottom="on", length=5, labelsize=8)
ax2.tick_params(axis='y',which='major', direction="out", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='y',which='minor', direction="out", top="off", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("Actividad1_Lorentziano.png", format="png",dpi=1000)

#Se define la función para un único pico con perfil de Voigt para separar cada pico en la figura
def _1Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )

#Se define la función para 4 picos con ajuste con perfil de voigt
def _4Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1, ampG2, cenG2, sigmaG2, ampL2, cenL2, widL2,
            ampG3, cenG3, sigmaG3, ampL3, cenL3, widL3, ampG4, cenG4, sigmaG4, ampL4, cenL4, widL4):
    return _1Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1) +\
           _1Voigt(x, ampG2, cenG2, sigmaG2, ampL2, cenL2, widL2) +\
           _1Voigt(x, ampG3, cenG3, sigmaG3, ampL3, cenL3, widL3) +\
           _1Voigt(x, ampG4, cenG4, sigmaG4, ampL4, cenL4, widL4)

#Se usa la gráfica de intensidad respecto al ángulo para analizar una aproximación 
#visual a los parámetros
ampG1 = 1
cenG1 = 17.5
sigmaG1 = 1
ampL1 = 1765
cenL1 = 17.5
widL1 = 1

ampG2 = 1
cenG2 = 19.8
sigmaG2 = 1
ampL2 = 6636
cenL2 = 19.8
widL2 = 1

ampG3 = 0.2
cenG3 = 41
sigmaG3 = 1
ampL3 = 411
cenL3 = 41
widL3 = 1

ampG4 = 2
cenG4 = 47.1
sigmaG4 = 5
ampL4 = 1261
cenL4 = 47.1
widL4 = 1

indices_voigt1 = np.where((angulo >= 17.0) & (angulo <= 18.0))
angulo_v1 = angulo[indices_voigt1]
intensidad_v1 = intensidad[indices_voigt1]

indices_voigt2 = np.where((angulo >= 19.0) & (angulo <= 20.5))
angulo_v2 = angulo[indices_voigt2]
intensidad_v2 = intensidad[indices_voigt2]

indices_voigt3 = np.where((angulo >= 40.5) & (angulo <= 41.4))
angulo_v3 = angulo[indices_voigt3]
intensidad_v3 = intensidad[indices_voigt3]

indices_voigt4 = np.where((angulo >= 46.5) & (angulo <= 47.7))
angulo_v4 = angulo[indices_voigt4]
intensidad_v4 = intensidad[indices_voigt4]

#Se calculan por separado los parámetros para el ajuste con perfil de Voigt de cada pico

#Cálculo de parámetros pico 1
popt_voigt1, pcov_voigt1 = curve_fit(_1Voigt, angulo_v1, intensidad_v1, p0=[ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1])

perr_1voigt = np.sqrt(np.diag(pcov_voigt1))

pars_1 = popt_voigt1[0:6]
voigt_peak_1 = _1Voigt(angulo, *pars_1)

#Cálculo de parámetros pico 2
popt_voigt2, pcov_voigt2 = curve_fit(_1Voigt, angulo_v2, intensidad_v2, p0=[ampG2, cenG2, sigmaG2, ampL2, cenL2, widL2])

perr_2voigt = np.sqrt(np.diag(pcov_voigt2))

pars_2 = popt_voigt2[0:6]
voigt_peak_2 = _1Voigt(angulo, *pars_2)

#Cálculo de parámetros pico 3
popt_voigt3, pcov_voigt3 = curve_fit(_1Voigt, angulo_v3, intensidad_v3, p0=[ampG3, cenG3, sigmaG3, ampL3, cenL3, widL3], maxfev=20000)

perr_3voigt = np.sqrt(np.diag(pcov_voigt3))

pars_3 = popt_voigt3[0:6]
voigt_peak_3 = _1Voigt(angulo, *pars_3)

#Cálculo de parámetros pico 4
popt_voigt4, pcov_voigt4 = curve_fit(_1Voigt, angulo_v4, intensidad_v4, p0=[ampG4, cenG4, sigmaG4, ampL4, cenL4, widL4])

perr_4voigt = np.sqrt(np.diag(pcov_voigt4))

pars_4 = popt_voigt4[0:6]
voigt_peak_4 = _1Voigt(angulo, *pars_4)

#Se hallan los residuales para el ajuste obtenido, tomando solo aquellos
#significativos para los picos
popt_voigt = np.concatenate((popt_voigt1,popt_voigt2,popt_voigt3,popt_voigt4),axis=None)
residual_4voigt = intensidad - (_4Voigt(angulo, *popt_voigt))
indices_residuales = np.where((angulo < 10))
residual_4voigt[indices_residuales] = np.nan

#Parámetros del ajuste junto con los errores de cada uno
print("-------------Peak 1-------------")
print("amplitude gaussian = %0.2f (+/-) %0.2f" % (pars_1[0], perr_1voigt[0]))
print("center gaussian = %0.2f (+/-) %0.2f" % (pars_1[1], perr_1voigt[1]))
print("sigma gaussian = %0.2f (+/-) %0.2f" % (pars_1[2], perr_1voigt[2]))
print("amplitude lorentzian = %0.2f (+/-) %0.2f" % (pars_1[3], perr_1voigt[3]))
print("center lorentzian = %0.2f (+/-) %0.2f" % (pars_1[4], perr_1voigt[4]))
print("width lorentzian = %0.2f (+/-) %0.2f" % (pars_1[5], perr_1voigt[5]))
print("gauss weight = %0.2f" % ((pars_1[0]/(pars_1[0]+pars_1[3]))*100))
print("lorentz weight = %0.2f" % ((pars_1[3]/(pars_1[0]+pars_1[3]))*100))
print("area = %0.2f" % np.trapz(voigt_peak_1))
print("--------------------------------")
print("-------------Peak 2-------------")
print("amplitude gaussian = %0.2f (+/-) %0.2f" % (pars_2[0], perr_2voigt[0]))
print("center gaussian = %0.2f (+/-) %0.2f" % (pars_2[1], perr_2voigt[1]))
print("sigma gaussian = %0.2f (+/-) %0.2f" % (pars_2[2], perr_2voigt[2]))
print("amplitude lorentzian = %0.2f (+/-) %0.2f" % (pars_2[3], perr_2voigt[3]))
print("center lorentzian = %0.2f (+/-) %0.2f" % (pars_2[4], perr_2voigt[4]))
print("width lorentzian = %0.2f (+/-) %0.2f" % (pars_2[5], perr_2voigt[5]))
print("gauss weight = %0.2f" % ((pars_2[0]/(pars_2[0]+pars_2[3]))*100))
print("lorentz weight = %0.2f" % ((pars_2[3]/(pars_2[0]+pars_2[3]))*100))
print("area = %0.2f" % np.trapz(voigt_peak_2))
print("--------------------------------")
print("-------------Peak 3-------------")
print("amplitude gaussian = %0.2f (+/-) %0.2f" % (pars_3[0], perr_3voigt[0]))
print("center gaussian = %0.2f (+/-) %0.2f" % (pars_3[1], perr_3voigt[1]))
print("sigma gaussian = %0.2f (+/-) %0.2f" % (pars_3[2], perr_3voigt[2]))
print("amplitude lorentzian = %0.2f (+/-) %0.2f" % (pars_3[3], perr_3voigt[3]))
print("center lorentzian = %0.2f (+/-) %0.2f" % (pars_3[4], perr_3voigt[4]))
print("width lorentzian = %0.2f (+/-) %0.2f" % (pars_3[5], perr_3voigt[5]))
print("gauss weight = %0.2f" % ((pars_3[0]/(pars_3[0]+pars_3[3]))*100))
print("lorentz weight = %0.2f" % ((pars_3[3]/(pars_3[0]+pars_3[3]))*100))
print("area = %0.2f" % np.trapz(voigt_peak_3))
print("--------------------------------")
print("-------------Peak 4-------------")
print("amplitude gaussian = %0.2f (+/-) %0.2f" % (pars_4[0], perr_4voigt[0]))
print("center gaussian = %0.2f (+/-) %0.2f" % (pars_4[1], perr_4voigt[1]))
print("sigma gaussian = %0.2f (+/-) %0.2f" % (pars_4[2], perr_4voigt[2]))
print("amplitude lorentzian = %0.2f (+/-) %0.2f" % (pars_4[3], perr_4voigt[3]))
print("center lorentzian = %0.2f (+/-) %0.2f" % (pars_4[4], perr_4voigt[4]))
print("width lorentzian = %0.2f (+/-) %0.2f" % (pars_4[5], perr_4voigt[5]))
print("gauss weight = %0.2f" % ((pars_4[0]/(pars_4[0]+pars_4[3]))*100))
print("lorentz weight = %0.2f" % ((pars_4[3]/(pars_4[0]+pars_4[3]))*100))
print("area = %0.2f" % np.trapz(voigt_peak_4))
print("--------------------------------")

#Se obtiene la gráfica para los datos arrojados
fig = plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
gs.update(hspace=0) 

ax1.plot(angulo, intensidad, "ro", markersize=3)
ax1.plot(angulo, _4Voigt(angulo, *popt_voigt), 'k--')#,\
         #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))
'''
# peak 1
ax1.plot(angulo, lorentz_peak_1, "g")
ax1.fill_between(angulo, lorentz_peak_1.min(), lorentz_peak_1, facecolor="green", alpha=0.5)
  
# peak 2
ax1.plot(angulo, lorentz_peak_2, "y")
ax1.fill_between(angulo, lorentz_peak_2.min(), lorentz_peak_2, facecolor="yellow", alpha=0.5)  

# peak 3
ax1.plot(angulo, lorentz_peak_3, "m")
ax1.fill_between(angulo, lorentz_peak_3.min(), lorentz_peak_3, facecolor="purple", alpha=0.5) 

# peak 4
ax1.plot(angulo, lorentz_peak_4, "c")
ax1.fill_between(angulo, lorentz_peak_4.min(), lorentz_peak_4, facecolor="cyan", alpha=0.5) 
'''

#Plot para residuales
ax2.plot(angulo, residual_4voigt, "bo", markersize=3)
    
#ax1.set_xlim(-5,105)
#ax1.set_ylim(-0.5,8)

#ax2.set_xlim(17,18)
#ax2.set_ylim(-0.5,0.75)

ax2.set_xlabel("Ángulo $\\theta$ (°)",family="serif",  fontsize=12)
ax1.set_ylabel("Intensidad - R 35kV - (Imp/s)",family="serif",  fontsize=12)
ax2.set_ylabel("Res.",family="serif",  fontsize=12)

ax1.legend(loc="best")

ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.xaxis.set_major_formatter(plt.NullFormatter())

ax1.tick_params(axis='x',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='x',which='minor', direction="out", top="on", right="on", bottom="off", length=5, labelsize=8)
ax1.tick_params(axis='y',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='y',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

ax2.tick_params(axis='x',which='major', direction="out", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='x',which='minor', direction="out", top="off", right="on", bottom="on", length=5, labelsize=8)
ax2.tick_params(axis='y',which='major', direction="out", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='y',which='minor', direction="out", top="off", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("Actividad1_PerfilVoigt.png", format="png",dpi=1000)

