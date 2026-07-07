import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
from numba import njit, prange
import threading
import random
import time

# NUMBA

@njit(parallel=True, fastmath=True)
def _calcular_julia_jit(cr, ci, xmin, xmax, ymin, ymax,
                         ancho, alto, max_iter):
    
    resultado = np.zeros((alto, ancho), dtype=np.float64)

    dx = (xmax - xmin) / ancho
    dy = (ymax - ymin) / alto

    for py in prange(alto):          
        y0 = ymin + py * dy          
        for px in range(ancho):
            x0 = xmin + px * dx      

            zr = x0
            zi = y0
            n  = 0

            while n < max_iter:
                zr2 = zr * zr
                zi2 = zi * zi

                if zr2 + zi2 > 4.0:          
                    log_mod = 0.5 * np.log(zr2 + zi2)
                    smooth  = n + 1.0 - np.log(log_mod) / np.log(2.0)
                    resultado[py, px] = smooth
                    break

                zi = 2.0 * zr * zi + ci      
                zr = zr2 - zi2 + cr          
                n += 1
    return resultado

#  Compilación de Numba

def _precompilar():
    """Lanza una compilación tiny al arrancar para que el primer zoom sea inmediato."""
    print("  Compilando núcleo matemático con Numba... ", end='', flush=True)
    _calcular_julia_jit(0.0, 0.0, -2.0, 2.0, -1.5, 1.5, 64, 50, 20)
    print("listo.")

_precompilar()


# Pasadas

PASADAS = [
    (160, 125,  40, 0.05),    # pasada 1: resolución baja, se ve al instante
    (400, 312,  80, 0.12),    # pasada 2: resolución media
    (800, 625, 160, None),    # pasada 3: resolución completa
]

def calcular_progresivo(c, xmin, xmax, ymin, ymax,
                        max_iter, callback, cancelado):
    """
    Calcula el Julia en 3 pasadas de resolución creciente.
    Después de cada pasada llama a callback(datos_normalizados).
    Si cancelado() devuelve True, aborta (el usuario movió la rueda de nuevo).
    """
    for ancho, alto, iters, pausa in PASADAS:
        if cancelado():
            return

        datos = _calcular_julia_jit(
            c.real, c.imag,
            xmin, xmax, ymin, ymax,
            ancho, alto,
            min(iters, max_iter)
        )

        if cancelado():
            return

        # Normalizar manteniendo el negro para el conjunto
        dentro = datos == 0.0
        fuera  = ~dentro
        norm   = np.zeros_like(datos)
        if fuera.any():
            v0, v1 = datos[fuera].min(), datos[fuera].max()
            if v1 > v0:
                norm[fuera] = (datos[fuera] - v0) / (v1 - v0)

        callback(norm, ancho, alto)

        if pausa and not cancelado():
            time.sleep(pausa)


# ─────────────────────────────────────────────────────────────────────────────
#  Paletas de color
# ─────────────────────────────────────────────────────────────────────────────

_PALETAS_DEF = {
    'cosmos':  [(0,'#000000'),(0.01,'#0D0221'),(0.08,'#1A0A5C'),
                (0.20,'#0047AB'),(0.40,'#00BFFF'),(0.65,'#E0F7FA'),
                (0.85,'#FFD700'),(1.00,'#FFFFFF')],
    'fuego':   [(0,'#000000'),(0.01,'#1A0000'),(0.15,'#7B0000'),
                (0.35,'#FF0000'),(0.60,'#FF6600'),(0.80,'#FFD700'),
                (1.00,'#FFFFFF')],
    'aurora':  [(0,'#000000'),(0.02,'#001A10'),(0.15,'#003300'),
                (0.35,'#00AA44'),(0.60,'#88FFAA'),(0.80,'#CCFFEE'),
                (1.00,'#FFFFFF')],
    'nebulosa':[(0,'#000000'),(0.02,'#1A001A'),(0.15,'#4B0082'),
                (0.35,'#8B008B'),(0.55,'#DA70D6'),(0.75,'#FFB6C1'),
                (1.00,'#FFFFFF')],
    'oceano':  [(0,'#000000'),(0.02,'#000820'),(0.15,'#003060'),
                (0.35,'#0077B6'),(0.60,'#00B4D8'),(0.80,'#90E0EF'),
                (1.00,'#FFFFFF')],
}
NOMBRES_PALETA = list(_PALETAS_DEF.keys())

def crear_paleta(nombre):
    datos = _PALETAS_DEF.get(nombre, _PALETAS_DEF['cosmos'])
    return mcolors.LinearSegmentedColormap.from_list(
        nombre, [(d[0], d[1]) for d in datos])

#  Valores de C que da Julia más interesantes (algunos de Bellos, otros aleatorios)

C_BELLOS = [
    complex(-0.7269,  0.1889),
    complex(-0.4,     0.6),
    complex( 0.285,   0.01),
    complex(-0.8,     0.156),
    complex(-0.7,     0.27015),
    complex( 0.0,     0.8),
    complex(-0.835,  -0.2321),
    complex(-0.1,     0.651),
    complex( 0.37,    0.1),
    complex(-1.476,   0.0),
    complex( 0.3,    -0.5),
    complex(-0.54,    0.54),
    complex(-0.624,   0.435),
    complex( 0.255,   0.0),
    complex(-0.75,    0.11),
]

#  Explorador

class ExploradorJulia:

    # ── Parámetros de zoom ───────────────────────────────────────
    ZOOM_FACTOR_RUEDA = 0.82    # cada tick de rueda = ×0.82 (hacia adentro)
    DEBOUNCE_MS       = 140     # ms de quietud antes de lanzar render completo
    MAX_ITER_INI      = 150
    MAX_ITER_MAX      = 900
    RES_W, RES_H      = 800, 620

    def __init__(self):
        # Estado matemático
        self.c          = random.choice(C_BELLOS)
        self.cx, self.cy = 0.0, 0.0
        self.zoom       = 3.2
        self.max_iter   = self.MAX_ITER_INI
        self.idx_paleta = 0
        self.idx_c      = 0

        # Estado del render en hilo secundario
        self._render_id   = 0        # incrementar cancela el render anterior
        self._render_lock = threading.Lock()
        self._ultimo_norm = None     # caché de la última matriz normalizada

        # Debounce de rueda
        self._zoom_acum    = 0.0     # zoom acumulado sin renderizar
        self._zoom_cx      = 0.0     # punto focal del zoom
        self._zoom_cy      = 0.0
        self._debounce_tmr = None

        # ── Construir ventana ─────────────────────────────────────
        self.fig = plt.figure(
            figsize=(13, 8.5),
            facecolor='#080810',
            num='Explorador de Conjuntos de Julia'
        )

        # Área de imagen
        self.ax = self.fig.add_axes([0.0, 0.08, 0.75, 0.92])
        self.ax.set_facecolor('#000000')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Panel de info
        self.ax_info = self.fig.add_axes([0.76, 0.35, 0.23, 0.63])
        self.ax_info.set_facecolor('#0D0D1A')
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        for sp in self.ax_info.spines.values():
            sp.set_edgecolor('#2A2A4A')

        # Botones
        bs = {'color': '#1A1A2E', 'hovercolor': '#2A2A4A'}
        self._make_btn([0.765, 0.27, 0.215, 0.055],
                       '⟳ Nuevo Julia  [ESPACIO]', bs, self.nuevo_julia)
        self._make_btn([0.765, 0.21, 0.215, 0.055],
                       '⌂ Resetear vista  [R]',    bs, self.reset_vista)
        self._make_btn([0.765, 0.15, 0.215, 0.055],
                       '◈ Cambiar color  [P]',     bs, self.siguiente_paleta)
        self._make_btn([0.765, 0.09, 0.215, 0.055],
                       '↓ Guardar PNG  [S]',       bs, self.guardar)

        # Barra de estado
        self.ax_st = self.fig.add_axes([0.0, 0.0, 1.0, 0.07])
        self.ax_st.set_facecolor('#0A0A15')
        self.ax_st.set_xticks([])
        self.ax_st.set_yticks([])
        for sp in self.ax_st.spines.values():
            sp.set_edgecolor('#1A1A2A')
        self.txt_st = self.ax_st.text(
            0.5, 0.5,
            'Rueda: zoom  |  Clic: centrar  |  ESPACIO: nuevo Julia'
            '  |  R: reset  |  S: guardar  |  P: paleta  |  +/-: detalle',
            ha='center', va='center', color='#555577', fontsize=8.5,
            transform=self.ax_st.transAxes
        )

        self.im = None

        # Eventos
        self.fig.canvas.mpl_connect('scroll_event',      self._ev_scroll)
        self.fig.canvas.mpl_connect('button_press_event',self._ev_click)
        self.fig.canvas.mpl_connect('key_press_event',   self._ev_key)

        self._lanzar_render()
        plt.show()

    # ── Helpers ───────────────────────────────────────────────────

    def _make_btn(self, rect, label, style, cb):
        ax  = self.fig.add_axes(rect)
        btn = Button(ax, label, **style)
        btn.label.set_color('#AAAACC')
        btn.label.set_fontsize(8.5)
        btn.on_clicked(lambda e: cb())
        return btn

    def _limites(self):
        asp = self.RES_H / self.RES_W
        hw  = self.zoom / 2
        hh  = self.zoom * asp / 2
        return self.cx-hw, self.cx+hw, self.cy-hh, self.cy+hh

    def _set_status(self, txt):
        self.txt_st.set_text(txt)
        self.fig.canvas.draw_idle()

    # ── Render en hilo secundario ─────────────────────────────────

    def _lanzar_render(self):
        """Cancela cualquier render en curso y lanza uno nuevo en un hilo."""
        with self._render_lock:
            self._render_id += 1
            rid = self._render_id

        c    = self.c
        xmin, xmax, ymin, ymax = self._limites()
        mi   = self.max_iter

        def cancelado():
            return self._render_id != rid

        def callback(norm, w, h):
            if cancelado():
                return
            # Actualizar la imagen desde el hilo principal via call_soon
            self.fig.canvas.flush_events()
            self._mostrar(norm, xmin, xmax, ymin, ymax)

        hilo = threading.Thread(
            target=calcular_progresivo,
            args=(c, xmin, xmax, ymin, ymax, mi, callback, cancelado),
            daemon=True
        )
        hilo.start()

    def _mostrar(self, norm, xmin, xmax, ymin, ymax):
        """Actualiza la imagen en el eje (llamado desde hilo secundario)."""
        self._ultimo_norm = norm
        cmap = crear_paleta(NOMBRES_PALETA[self.idx_paleta])

        if self.im is None:
            self.im = self.ax.imshow(
                norm,
                extent=[xmin, xmax, ymin, ymax],
                origin='lower', cmap=cmap,
                vmin=0.0, vmax=1.0,
                interpolation='bilinear',
                aspect='auto'
            )
        else:
            self.im.set_data(norm)
            self.im.set_extent([xmin, xmax, ymin, ymax])
            self.im.set_cmap(cmap)

        self._actualizar_info(xmin, xmax, ymin, ymax)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _actualizar_info(self, xmin, xmax, ymin, ymax):
        self.ax_info.clear()
        self.ax_info.set_facecolor('#0D0D1A')
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        for sp in self.ax_info.spines.values():
            sp.set_edgecolor('#2A2A4A')

        mod_c   = abs(self.c)
        arg_c   = np.degrees(np.angle(self.c))
        zoom_x  = 3.2 / self.zoom

        lineas = [
            ('CONJUNTO DE JULIA', '#8888FF', 10, 'bold'),
            ('', None, 6, 'normal'),
            ('Parámetro c:', '#AAAACC', 8.5, 'bold'),
            (f'  {self.c.real:+.7f}', '#EEEEFF', 8.5, 'normal'),
            (f'  {self.c.imag:+.7f}i', '#EEEEFF', 8.5, 'normal'),
            ('', None, 6, 'normal'),
            ('Forma polar (Moivre):', '#AAAACC', 8.5, 'bold'),
            (f'  |c| = {mod_c:.6f}', '#CCDDFF', 8.5, 'normal'),
            (f'  arg = {arg_c:.3f}°', '#CCDDFF', 8.5, 'normal'),
            ('', None, 6, 'normal'),
            ('Vista:', '#AAAACC', 8.5, 'bold'),
            (f'  Re [{xmin:.5f}, {xmax:.5f}]', '#BBCCEE', 7.5, 'normal'),
            (f'  Im [{ymin:.5f}, {ymax:.5f}]', '#BBCCEE', 7.5, 'normal'),
            (f'  Zoom ×{zoom_x:.1f}', '#BBCCEE', 8, 'normal'),
            ('', None, 6, 'normal'),
            (f'Iteraciones: {self.max_iter}', '#AAAACC', 8.5, 'bold'),
            (f'Paleta: {NOMBRES_PALETA[self.idx_paleta].upper()}', '#AAAACC', 8.5, 'bold'),
            ('', None, 6, 'normal'),
            ('─'*22, '#2A2A4A', 7, 'normal'),
            ('', None, 4, 'normal'),
            ('REGLA MATEMÁTICA:', '#AAAACC', 8, 'bold'),
            ('z₀ = píxel del plano', '#777788', 7.5, 'normal'),
            ('z_{n+1} = zₙ² + c', '#777788', 7.5, 'normal'),
            ('Negro: |z| nunca > 2', '#777788', 7.5, 'normal'),
        ]

        y = 0.97
        for txt, col, sz, w in lineas:
            if col is None:
                y -= 0.012
                continue
            self.ax_info.text(0.05, y, txt, transform=self.ax_info.transAxes,
                              color=col, fontsize=sz, fontweight=w,
                              va='top', fontfamily='monospace')
            y -= 0.030 if sz >= 9 else 0.024

    # ── Debounce de zoom ──────────────────────────────────────────

    def _aplicar_zoom_acumulado(self):
        """
        Se llama cuando el usuario deja de mover la rueda.
        Aplica el zoom total acumulado y lanza el render de alta calidad.
        """
        if self._zoom_acum == 0.0:
            return

        factor = self._zoom_acum
        # Centrar el zoom en el punto focal acumulado
        self.cx   = self._zoom_cx + (self.cx - self._zoom_cx) * factor
        self.cy   = self._zoom_cy + (self.cy - self._zoom_cy) * factor
        self.zoom *= factor
        self.zoom  = max(1e-10, min(self.zoom, 6.0))

        self._zoom_acum = 0.0
        self._lanzar_render()

    def _programar_debounce(self):
        """Reinicia el temporizador de debounce cada vez que llega un evento de rueda."""
        if self._debounce_tmr is not None:
            try:
                self._debounce_tmr.cancel()
            except Exception:
                pass
        self._debounce_tmr = threading.Timer(
            self.DEBOUNCE_MS / 1000.0,
            self._aplicar_zoom_acumulado
        )
        self._debounce_tmr.daemon = True
        self._debounce_tmr.start()

    # ── Eventos ───────────────────────────────────────────────────

    def _ev_scroll(self, event):
        if event.inaxes != self.ax:
            return

        # Factor de zoom de este tick
        tick = self.ZOOM_FACTOR_RUEDA if event.button == 'up' else (1.0 / self.ZOOM_FACTOR_RUEDA)

        # Punto focal bajo el cursor (en coordenadas del plano complejo)
        if event.xdata is not None and event.ydata is not None:
            px, py = event.xdata, event.ydata
        else:
            px, py = self.cx, self.cy

        # Primera vez del gesto: fijar punto focal
        if self._zoom_acum == 0.0:
            self._zoom_cx = px
            self._zoom_cy = py
            self._zoom_acum = tick
        else:
            # Acumular factor (multiplicativo)
            self._zoom_acum *= tick
            # Promedio ponderado del punto focal (para gestos que se mueven)
            self._zoom_cx = 0.7 * self._zoom_cx + 0.3 * px
            self._zoom_cy = 0.7 * self._zoom_cy + 0.3 * py

        # Vista previa instantánea: escalar la imagen existente sin recalcular
        if self.im is not None:
            xmin, xmax, ymin, ymax = self._limites()
            hw = (xmax - xmin) * self._zoom_acum / 2
            hh = (ymax - ymin) * self._zoom_acum / 2
            ncx = self._zoom_cx + (self.cx - self._zoom_cx) * self._zoom_acum
            ncy = self._zoom_cy + (self.cy - self._zoom_cy) * self._zoom_acum
            self.im.set_extent([ncx-hw, ncx+hw, ncy-hh, ncy+hh])
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass

        self._programar_debounce()

    def _ev_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None:
            return
        self.cx   = event.xdata
        self.cy   = event.ydata
        self.zoom *= 0.45
        self._lanzar_render()

    def _ev_key(self, event):
        k = event.key
        if   k == ' ':         self.nuevo_julia()
        elif k in ('r','R'):   self.reset_vista()
        elif k in ('s','S'):   self.guardar()
        elif k in ('p','P'):   self.siguiente_paleta()
        elif k in ('+','='):
            self.max_iter = min(self.max_iter + 40, self.MAX_ITER_MAX)
            self._lanzar_render()
        elif k == '-':
            self.max_iter = max(self.max_iter - 40, 30)
            self._lanzar_render()
        elif k in ('q','Q','escape'):
            plt.close('all')

    # ── Acciones ─────────────────────────────────────────────────

    def nuevo_julia(self):
        self.idx_c = (self.idx_c + 1) % len(C_BELLOS)
        if random.random() < 0.30:
            r = random.uniform(0.25, 0.95)
            θ = random.uniform(0, 2 * np.pi)
            self.c = complex(r * np.cos(θ), r * np.sin(θ))
        else:
            self.c = C_BELLOS[self.idx_c]
        self.reset_vista()

    def reset_vista(self):
        self.cx, self.cy = 0.0, 0.0
        self.zoom = 3.2
        self._lanzar_render()

    def siguiente_paleta(self):
        self.idx_paleta = (self.idx_paleta + 1) % len(NOMBRES_PALETA)
        # Recolorea sin recalcular (usa el caché)
        if self._ultimo_norm is not None:
            xmin, xmax, ymin, ymax = self._limites()
            self._mostrar(self._ultimo_norm, xmin, xmax, ymin, ymax)
        else:
            self._lanzar_render()

    def guardar(self):
        nombre = (f'julia_{self.c.real:+.5f}_{self.c.imag:+.5f}'
                  f'_z{self.zoom:.5f}.png').replace('+','p').replace('-','n')
        xmin, xmax, ymin, ymax = self._limites()
        print(f'\n  Guardando {nombre}...')
        datos = _calcular_julia_jit(
            self.c.real, self.c.imag,
            xmin, xmax, ymin, ymax,
            1400, 1100,
            min(self.max_iter * 2, self.MAX_ITER_MAX)
        )
        dentro = datos == 0.0
        fuera  = ~dentro
        norm   = np.zeros_like(datos)
        if fuera.any():
            v0, v1 = datos[fuera].min(), datos[fuera].max()
            if v1 > v0:
                norm[fuera] = (datos[fuera] - v0) / (v1 - v0)
        cmap = crear_paleta(NOMBRES_PALETA[self.idx_paleta])
        fig2, ax2 = plt.subplots(figsize=(14, 11), dpi=160)
        fig2.patch.set_facecolor('#000000')
        ax2.set_facecolor('#000000')
        ax2.imshow(norm, extent=[xmin,xmax,ymin,ymax], origin='lower',
                   cmap=cmap, vmin=0, vmax=1, interpolation='bilinear', aspect='auto')
        ax2.set_xlabel('Re(z)', color='white', fontsize=10)
        ax2.set_ylabel('Im(z)', color='white', fontsize=10)
        ax2.tick_params(colors='#555555')
        ax2.set_title(
            f'Conjunto de Julia  —  c = {self.c.real:+.7f} + {self.c.imag:+.7f}i\n'
            f'z_{{n+1}} = z_n² + c  |  Zoom ×{3.2/self.zoom:.0f}  |  {self.max_iter} iter.',
            color='white', fontsize=11)
        for sp in ax2.spines.values(): sp.set_edgecolor('#333333')
        plt.tight_layout()
        plt.savefig(nombre, dpi=160, bbox_inches='tight', facecolor='#000000')
        plt.close(fig2)
        print(f'  Guardado: {nombre}')
        self._set_status(f'✓ Guardado: {nombre}')


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  EXPLORADOR DE CONJUNTOS DE JULIA — VERSIÓN RÁPIDA")
    print("=" * 60)
    print()
    print("  Controles:")
    print("    Rueda          → Zoom fluido")
    print("    Clic izquierdo → Centrar y acercar")
    print("    ESPACIO        → Nuevo Julia aleatorio")
    print("    R              → Resetear vista")
    print("    S              → Guardar PNG (alta resolución)")
    print("    P              → Cambiar paleta (instantáneo)")
    print("    + / -          → Más / menos detalle")
    print("    Q / Escape     → Salir")
    print()

    ExploradorJulia()
