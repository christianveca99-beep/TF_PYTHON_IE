import numpy as np
from numpy.linalg import inv, eig
from math import *
import matplotlib.pyplot as plt
import pandas as pd

# ==========================
# Unidades MKS
# ==========================
m = 1
cm = 0.01 * m
mm = 0.001 * m

kgf = 1
tnf = 1000 * kgf


# ==========================
# Análisis Modal
# ==========================
class Analisis_Modal:
    def __init__(self, m, k, h):
        self.m = m
        self.k = k
        self.h = h

        self.M = self.Masa()
        self.K = self.K_general()
        self.w, self.T, self.x = self.Modos_formas()
        self.phi, self.I = self.Normalizacion(self.x)
        self.MPF, self.EM, self.PM = self.Factor_Part_Modal(self.phi, self.I)
        self.modes, self.results = self.Tablas(self.x, self.T, self.MPF, self.EM, self.PM)

    def Masa(self):
        n_mass = len(self.m)
        M = np.zeros((n_mass, n_mass))
        for i in range(n_mass):
            M[i][i] = round(self.m[i], 2)
        return M

    def K_general(self):
        n_k = len(self.k)
        K = np.zeros((n_k, n_k))
        K[0][0] = self.k[0]
        for i in range(1, n_k):
            K[i - 1:i + 1, i - 1:i + 1] += np.array([[self.k[i], -self.k[i]],
                                                       [-self.k[i], self.k[i]]])
        return K

    def Modos_formas(self):
        A = inv(self.M) @ self.K
        eigenvalues, eigenvectors = eig(A)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        n_mod = eigenvalues.shape[0]
        w = np.sqrt(eigenvalues)
        for i in range(n_mod - 1):
            for j in range(n_mod - 1 - i):
                if w[j] > w[j + 1]:
                    w[j], w[j + 1] = w[j + 1], w[j]
                    eigenvectors[:, [j, j + 1]] = eigenvectors[:, [j + 1, j]]
        T = [round(2 * pi / f, 2) for f in w]
        return w, T, eigenvectors

    def Normalizacion(self, eigenvectors):
        n_x = eigenvectors.shape[0]
        phi = np.zeros((n_x, n_x))
        I = n_x * [1]
        for i in range(n_x):
            x = eigenvectors[:, i]
            phi[:, i] = (1 / sqrt(x.T @ self.M @ x)) * x
        return phi, np.array(I)

    def Factor_Part_Modal(self, phi, I):
        R = []
        Efc_mass = []
        Part_mass = []
        for i in range(phi.shape[0]):
            r = phi[:, i].T @ self.M @ I.T
            R.append(round(r, 2))
            Efc_mass.append(round(r ** 2, 2))
        for i in range(len(R)):
            Part_mass.append(round(100 * Efc_mass[i] / sum(Efc_mass), 2))
        return R, Efc_mass, Part_mass

    def Graficos(self):
        n_gr = self.x.shape[0]
        basement = n_gr * [0]
        graph_modes = np.vstack((basement, self.x))
        hn = [0] + self.h
        for i in range(1, len(hn)):
            hn[i] += hn[i - 1]
        fig, axs = plt.subplots(1, n_gr, figsize=(20, 10))
        if n_gr == 1:
            axs = [axs]
        for i in range(n_gr):
            axs[i].plot(graph_modes[:, i], hn, 'b-o', (n_gr + 1) * [0], hn, 'black')
            axs[i].grid('silver')
            axs[i].set_xlim([-1, 1])
            axs[i].set_ylim([0, max(hn)])
            axs[i].set_title(f'Modo {i + 1} (T{i + 1} = {self.T[i]}s)', size=14)
        return fig

    def Tablas(self, modes, T, MPF, EM, PM):
        fmode, cmode = modes.shape
        n_mode = [("Modo " + str(i + 1)) for i in range(modes.shape[1])]
        story = {i: ("NIVEL " + str(fmode - i)) for i in range(fmode)}
        df_modes = pd.DataFrame(np.around(modes[::-1], 4), columns=n_mode)
        df_modes.rename(index=story, inplace=True)

        mode_col = [i + 1 for i in range(cmode)]
        df_results = pd.DataFrame({"Modo": mode_col,
                                    "Periodo (s)": T,
                                    "Factor Part. masa": MPF,
                                    "Masa Efectiva": EM,
                                    "Masa Participativa (%)": PM})
        df_results.index = [''] * len(df_results)
        return df_modes, df_results


# ==========================
# Análisis Espectral
# ==========================
class Analisis_Espectral(Analisis_Modal):
    def __init__(self, m, k, h, S_coeff, R, Tp, Tl):
        super().__init__(m, k, h)
        self.S_coeff = S_coeff
        self.R = R
        self.Tp = Tp
        self.Tl = Tl

        self.Sa, self.A = self.Aceleraciones(self.T, self.MPF, self.phi)
        self.Sd, self.D, self.Dr = self.Desplazamientos(self.w, self.Sa, self.MPF, self.phi)
        self.F, self.Fr = self.Fuerzas(self.M, self.A)
        self.V, self.Vr = self.Cortantes(self.F)

    def Aceleraciones(self, T_x, MPF, phi):
        Sa = []
        n_acc = len(T_x)
        A = np.zeros((n_acc, n_acc))
        for i in range(n_acc):
            if T_x[i] < 0.2 * self.Tp:
                C = 1 + 7.5 * T_x[i] / self.Tp  # Adicion de nuevo rango segun la Norma E030 2026
            elif 0.2 * self.Tp <= T_x[i] <= self.Tp:
                C = 2.5
            elif self.Tp < T_x[i] < self.Tl:
                C = 2.5 * self.Tp / T_x[i]
            else:
                C = 2.5 * (self.Tp * self.Tl) / (T_x[i] ** 2)
            Sa.append(self.S_coeff * C)
            A[:, i] = Sa[i] * MPF[i] * phi[:, i]
        return Sa, A

    def Desplazamientos(self, w, Sa, MPF, phi):
        Sd = []
        n_disp = len(Sa)
        D = np.zeros((n_disp, n_disp))
        for i in range(n_disp):
            Sd.append(Sa[i] / (w[i] ** 2))
            D[:, i] = Sd[i] * MPF[i] * phi[:, i] / cm
        Dr = self.Combinacion_Modal(D)
        return Sd, D, Dr

    def Fuerzas(self, M, A):
        F = M @ A * (1 / tnf)
        Fr = self.Combinacion_Modal(F)
        return F, Fr

    def Cortantes(self, F):
        n_V = F.shape[0]
        V = np.zeros((n_V, n_V))
        V[0, :] = F[0, :]
        for i in range(1, n_V):
            V[i, :] = V[i - 1, :] + F[i, :]
        Vr = self.Combinacion_Modal(V)
        return V, Vr

    def Combinacion_Modal(self, X):
        n_arr = X.shape[0]
        Abs = np.array((n_arr) * [0], dtype=np.float64)
        Sqr = np.array((n_arr) * [0], dtype=np.float64)
        for i in range(n_arr):
            Abs += np.abs(X[:, i])
            Sqr += np.square(X[:, i])
        Sqr = np.sqrt(Sqr)
        r = 0.25 * Abs + 0.75 * Sqr
        return r

    def Graficos(self):
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2])

        n_gr = self.x.shape[0]
        desp_modes = np.vstack((n_gr * [0], self.D))
        hn = [0] + self.h
        legend_ax1 = []
        for i in range(1, len(hn)):
            hn[i] += hn[i - 1]
        ax1 = fig.add_subplot(gs[0])
        for i in range(n_gr):
            ax1.plot(desp_modes[:, i], hn, '-o')
            modes_max = np.round(np.max(desp_modes[:, i]), 2)
            legend_ax1.append('Δ max = ' + str(modes_max) + ' cm')
        ax1.set_title('Desplazamientos por cada modo (cm)', size=12)
        ax1.legend(legend_ax1, loc='lower right')
        ax1.grid("silver")
        ax1.set_ylim([0, max(hn)])

        desp_real = (0.75 * self.R) * np.hstack(([0], self.Dr))
        desp_max = np.round(np.max(desp_real), 2)
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(desp_real, hn, 'b-o', label='Δ max = ' + str(desp_max) + ' cm')
        ax2.set_title('Desplazamientos reales (cm)', size=12)
        ax2.legend(loc='lower right')
        ax2.grid("silver")
        ax2.set_ylim([0, max(hn)])

        drift = [0]
        for i in range(1, len(hn)):
            d = (desp_real[i] - desp_real[i - 1]) / ((hn[i] - hn[i - 1]) / cm) * 1000
            drift.append(d)
        drift_max = np.round(max(drift), 2)
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(drift, hn, 'b-o', label='Drift max = ' + str(drift_max) + ' (‰)')
        ax3.set_title('Derivas (‰)', size=12)
        ax3.legend(loc='lower right')
        ax3.grid("silver")
        ax3.set_ylim([0, max(hn)])

        stories = [i + 1 for i in range(self.Vr.shape[0])]
        ax4 = fig.add_subplot(gs[3])
        ax4.barh(stories, self.Vr[::-1])
        for index, value in enumerate(self.Vr[::-1]):
            ax4.text(value, index + 1, str(np.round(value, 2)), va='center', ha='left', fontsize=10)
        ax4.set_title('Cortantes reales (ton)', size=12)
        ax4.set_xlim([0, max(self.Vr) * 1.2])

        return fig


# ==========================
# Análisis Bidireccional E.030
# ==========================
class Analisis_Bidireccional:
    def __init__(self, analisis_x, analisis_y):
        self.eje_x = analisis_x
        self.eje_y = analisis_y
        self.h = analisis_x.h

        self.desplazamiento_x, self.desplazamiento_y, self.desplazamiento_bi = self.desplazamientos_combinados()
        self.drift_x, self.drift_y, self.drift_bi = self.Derivas_combinadas()
        self.V_x, self.V_y, self.V_bi = self.Cortantes_Combinadas()

    def _desplazamiento_real(self, analisis_modal_bi):
        return (0.75 * analisis_modal_bi.R) * np.hstack(([0], analisis_modal_bi.Dr))

    def desplazamientos_combinados(self):
        desplazamiento_x = self._desplazamiento_real(self.eje_x)
        desplazamiento_y = 0.30 * self._desplazamiento_real(self.eje_y)
        desplazamiento_bi = np.sqrt(desplazamiento_x ** 2 + desplazamiento_y ** 2)
        return desplazamiento_x, desplazamiento_y, desplazamiento_bi

    def _derivas(self, desplazamiento_real):
        hn = [0] + self.h
        for i in range(1, len(hn)):
            hn[i] += hn[i - 1]
        drift = [0]
        for i in range(1, len(hn)):
            d = (desplazamiento_real[i] - desplazamiento_real[i - 1]) / ((hn[i] - hn[i - 1]) / cm) * 1000
            drift.append(d)
        return np.array(drift)

    def Derivas_combinadas(self):
        drift_x = self._derivas(self.desplazamiento_x)
        drift_y = self._derivas(self.desplazamiento_y)
        drif_bi = np.sqrt(drift_x ** 2 + drift_y ** 2)
        return drift_x, drift_y, drif_bi

    def Cortantes_Combinadas(self):
        V_x = self.eje_x.Vr
        V_y = self.eje_y.Vr
        V_bi = np.sqrt(V_x ** 2 + V_y ** 2)
        return V_x, V_y, V_bi

    def Tabla_Resultados(self):
        desplazamiento_x = self.desplazamiento_x[1:]
        desplazamiento_y = self.desplazamiento_y[1:]
        desplazamiento_bi = self.desplazamiento_bi[1:]
        drift_x = self.drift_x[1:]
        drift_y = self.drift_y[1:]
        drift_bi = self.drift_bi[1:]

        n = len(drift_bi)
        niveles = [f"NIVEL {n - i}" for i in range(n)]
        df = pd.DataFrame({
            "Desp. X (cm)": np.round(desplazamiento_x[::-1], 2),
            "Desp. Y (cm)": np.round(desplazamiento_y[::-1], 2),
            "Desp. Combinado SRSS (cm)": np.round(desplazamiento_bi[::-1], 2),
            "Deriva X (‰)": np.round(drift_x[::-1], 4),
            "Deriva Y (‰)": np.round(drift_y[::-1], 4),
            "Deriva Combinada SRSS (‰)": np.round(drift_bi[::-1], 4),
        })
        df.index = niveles
        return df

    def Graficos_bidireccionales(self):
        hn = [0] + self.h
        for i in range(1, len(hn)):
            hn[i] += hn[i - 1]
        stories = [i + 1 for i in range(self.V_bi.shape[0])]

        fig, axs = plt.subplots(1, 3, figsize=(18, 8))

        desplazamiento_maximo_x = np.round(np.max(self.desplazamiento_x), 2)
        desplazamiento_maximo_y = np.round(np.max(self.desplazamiento_y), 2)
        desplazamiento_maximo_bi = np.round(np.max(self.desplazamiento_bi), 2)
        axs[0].plot(self.desplazamiento_x, hn, 'o-', label=f'Eje X (100% X)  = {desplazamiento_maximo_x} cm')
        axs[0].plot(self.desplazamiento_y, hn, 'o-', label=f'Eje Y (30% Y)  = {desplazamiento_maximo_y} cm')
        axs[0].plot(self.desplazamiento_bi, hn, 'k-o', linewidth=2, label=f'Combinacion  = {desplazamiento_maximo_bi} cm')
        axs[0].set_title('Desplazamientos reales bidireccionales (cm)', size=12)
        axs[0].legend(loc='lower right')
        axs[0].grid('silver')
        axs[0].set_ylim([0, max(hn)])

        drift_maxima_x = np.round(np.max(self.drift_x), 2)
        drift_maxima_y = np.round(np.max(self.drift_y), 2)
        drift_maxima_bi = np.round(np.max(self.drift_bi), 2)
        axs[1].plot(self.drift_x, hn, 'o-', label=f'Eje X  = {drift_maxima_x} ‰')
        axs[1].plot(self.drift_y, hn, 'o-', label=f'Eje Y  = {drift_maxima_y} ‰')
        axs[1].plot(self.drift_bi, hn, 'k-o', linewidth=2, label=f'Combinacion  = {drift_maxima_bi} ‰')
        axs[1].set_title('Derivas bidireccionales (‰)', size=12)
        axs[1].legend(loc='lower right')
        axs[1].grid('silver')
        axs[1].set_ylim([0, max(hn)])

        axs[2].barh(stories, self.V_bi[::-1], label='Cortante bi SRSS')
        for index, value in enumerate(self.V_bi[::-1]):
            axs[2].text(value, index + 1, str(np.round(value, 2)), va='center', ha='left', fontsize=10)
        axs[2].set_title('Cortantes de entrepiso bidireccionales (ton)', size=12)
        axs[2].set_xlim([0, max(self.V_bi) * 1.2])

        return fig
