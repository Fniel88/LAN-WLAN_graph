# final.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib.table import Table
from matplotlib.patches import Patch
import matplotlib.animation as animation

# ------------------------
# 1) DEFINICIÓN DE NODOS, IPS Y GRAFOS
# ------------------------
G = nx.Graph()

labels = {
    "ISP": "ISP\nFibra óptica",
    "Router": "Router\n192.168.1.1",
    "Switch-LAN": "Switch-LAN",
    "PC-Admin": "PC-Admin\n192.168.1.10",
    "Laptop-Admin": "Laptop-Admin\n192.168.1.11",
    "Impresora": "Impresora\n192.168.1.20",
    "Servidor HTTP": "Servidor HTTP\n192.168.1.100",
    "Access Point": "Access Point\n192.168.2.1",
    "Laptop Profesor": "Laptop Profesor\n192.168.2.10",
    "Tablet": "Tablet\n192.168.2.11",
    "Smartphone": "Smartphone\n192.168.2.12"
}

core_nodes = ["Router", "Switch-LAN", "ISP"]
lan_nodes = ["PC-Admin", "Laptop-Admin", "Impresora", "Servidor HTTP", "Access Point"]
wlan_nodes = ["Laptop Profesor", "Tablet", "Smartphone"]

# Conexiones LAN (UTP)
lan_edges = [
    ("Router", "Switch-LAN"),
    ("Switch-LAN", "PC-Admin"),
    ("Switch-LAN", "Laptop-Admin"),
    ("Switch-LAN", "Impresora"),
    ("Switch-LAN", "Servidor HTTP"),
    ("Switch-LAN", "Access Point")  # ← ahora el AP está conectado al switch
]

# Conexiones WLAN
wlan_edges = [
    ("Access Point", "Laptop Profesor"),
    ("Access Point", "Tablet"),
    ("Access Point", "Smartphone")
]

# Enlace de fibra óptica entre Router e ISP
fibra_edges = [("Router", "ISP")]

# Crear grafo
G.add_nodes_from(labels.keys())
G.add_edges_from(lan_edges + wlan_edges + fibra_edges)

# POSICIONES de los nodos (ajustadas visualmente)
pos = {
    "ISP": (1.3, 0.8),             # → movido a la derecha
    "Router": (0.0, 0.0),
    "Switch-LAN": (0.0, -0.8),
    "Servidor HTTP": (-0.8, -0.2),  # ← movido a la izquierda
    "PC-Admin": (-0.8, -1.6),
    "Laptop-Admin": (0.0, -1.6),
    "Impresora": (0.8, -1.6),
    "Access Point": (1.4, -0.4),
    "Laptop Profesor": (1.8, 0.5),
    "Tablet": (1.6, -0.9),
    "Smartphone": (2.3, -0.6)
}

# ------------------------
# 2) CODIFICACIÓN MANCHESTER
# ------------------------
mensaje = "HOLA"
binario = ''.join(format(ord(c), '08b') for c in mensaje)

def manchester(bits):
    samples = []
    for b in bits:
        samples.extend([1, -1] if b == '1' else [-1, 1])
    return np.array(samples, dtype=float)

senal = manchester(binario)

# FFT
N = len(senal)
yf = np.abs(fft(senal))
xf = fftfreq(N, d=1)

# ------------------------
# 3) DIBUJO COMPLETO
# ------------------------
fig = plt.figure(figsize=(20, 12))
grid = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0], height_ratios=[1.2, 1, 1])

ax1 = fig.add_subplot(grid[0:2, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[1, 1])
ax4 = fig.add_subplot(grid[2, :])

ax1.set_title("Topología de Red Híbrida con Conexión ISP (Fibra Óptica)", fontsize=14, fontweight="bold")

# Dibujar nodos
nx.draw_networkx_nodes(G, pos, nodelist=["ISP"], node_color="#f4d03f", node_size=2300, edgecolors="k", ax=ax1)
nx.draw_networkx_nodes(G, pos, nodelist=["Router", "Switch-LAN"], node_color="#e63946", node_size=2500, edgecolors="k", ax=ax1)
nx.draw_networkx_nodes(G, pos, nodelist=lan_nodes, node_color="#1d3557", node_size=2000, edgecolors="k", ax=ax1)
nx.draw_networkx_nodes(G, pos, nodelist=wlan_nodes, node_color="#2a9d8f", node_size=1800, edgecolors="k", ax=ax1)

# Dibujar enlaces
nx.draw_networkx_edges(G, pos, edgelist=fibra_edges, edge_color="#f1c40f", width=3.5, style='solid', ax=ax1)
nx.draw_networkx_edges(G, pos, edgelist=lan_edges, edge_color="#457b9d", width=2.2, style='solid', ax=ax1)
nx.draw_networkx_edges(G, pos, edgelist=wlan_edges, edge_color="#2b9348", width=2.0, style='dashed', ax=ax1)

# Etiquetas
for n, (x, y) in pos.items():
    ax1.text(x, y, labels[n], fontsize=9, ha="center", va="center",
             bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.4'),
             color='white')

ax1.axis('off')

# Leyenda
legend_patches = [
    Patch(facecolor="#f4d03f", edgecolor="k", label="ISP (Fibra Óptica)"),
    Patch(facecolor="#e63946", edgecolor="k", label="Nodos Core (Router/Switch)"),
    Patch(facecolor="#1d3557", edgecolor="k", label="LAN (UTP)"),
    Patch(facecolor="#2a9d8f", edgecolor="k", label="WLAN (Wireless)"),
    Patch(facecolor="#457b9d", edgecolor="#457b9d", label="Cable UTP"),
    Patch(facecolor="#2b9348", edgecolor="#2b9348", label="Enlace inalámbrico (WLAN, discontinua)"),
    Patch(facecolor="#f1c40f", edgecolor="#f1c40f", label="Enlace de Fibra Óptica")
]
ax1.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)

# Señal Manchester
ax2.step(np.arange(len(senal)), senal, where='mid', linewidth=2)
ax2.set_title("Señal Digital – Codificación Manchester", fontsize=12, fontweight="bold")
ax2.set_xlabel("Tiempo (muestras)")
ax2.set_ylabel("Amplitud")
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle="--", alpha=0.4)

bit_ticks = np.arange(0.5, len(senal)+0.5, 2)
bit_labels = list(binario)
ax2.set_xticks(bit_ticks)
ax2.set_xticklabels(bit_labels, fontsize=8)
ax2.annotate(f"Mensaje: '{mensaje}'  ->  Binario: {binario}", xy=(0.01, -1.3), xycoords='data', fontsize=9)

# Espectro FFT
half = N // 2
ax3.plot(xf[:half], yf[:half], linewidth=2.0, color='purple')
ax3.set_title("Espectro de Frecuencia (FFT)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Frecuencia (unidades relativas)")
ax3.set_ylabel("Magnitud")
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.fill_between(xf[:half], yf[:half], alpha=0.3, color='purple')

# Tabla de Enrutamiento
ax4.axis('off')
contenido = [
    ["Red / Máscara", "Gateway", "Interfaz", "Descripción"],
    ["192.168.1.0/24", "192.168.1.1", "LAN", "Segmento Administración"],
    ["192.168.2.0/24", "192.168.2.1", "WLAN", "Segmento Académico"],
    ["0.0.0.0/0", "ISP", "WAN", "Salida a Internet por Fibra"]
]

tabla = Table(ax4, bbox=[0.05, 0.1, 0.9, 0.8])
for i, fila in enumerate(contenido):
    for j, celda in enumerate(fila):
        if i == 0:
            facecolor = "#1d3557"; text_color = "white"; fontweight = "bold"; fontsize = 13
        else:
            facecolor = "#f1faee" if i % 2 == 1 else "#e9ecef"
            text_color = "black"; fontweight = "normal"; fontsize = 11
        cell = tabla.add_cell(i, j, width=0.25, height=0.25, 
                             text=celda, loc="center",
                             facecolor=facecolor, edgecolor='black')
        cell.get_text().set_fontsize(fontsize)
        cell.get_text().set_color(text_color)
        cell.get_text().set_fontweight(fontweight)
ax4.set_title("Tabla de Enrutamiento del Router", fontsize=14, fontweight="bold", pad=15)
ax4.add_table(tabla)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# -------------------------------
# ANIMACIÓN DEL MENSAJE "HOLA"
# -------------------------------
plt.savefig("topologia_final_mejorada.png", dpi=200, bbox_inches="tight")

# Nueva ruta: ISP → Router → Switch-LAN → Access Point → Laptop Profesor
ruta = ["ISP", "Router", "Switch-LAN", "Access Point", "Laptop Profesor"]
coords = [pos[n] for n in ruta]

mensaje_dot, = ax1.plot([], [], 'ro', markersize=14, zorder=5)
texto = ax1.text(coords[0][0], coords[0][1] + 0.25, "", fontsize=12, color="red", fontweight="bold")

def init():
    mensaje_dot.set_data([], [])
    texto.set_text("")
    return mensaje_dot, texto

def update(frame):
    salto = frame // 25
    progreso = (frame % 25) / 25.0
    if salto < len(coords) - 1:
        x = coords[salto][0] + progreso * (coords[salto + 1][0] - coords[salto][0])
        y = coords[salto][1] + progreso * (coords[salto + 1][1] - coords[salto][1])
    else:
        x, y = coords[-1]
    mensaje_dot.set_data([x], [y])
    texto.set_position((x, y + 0.25))
    if salto < len(coords) - 1:
        texto.set_text("Mensaje 'HOLA'")
    else:
        texto.set_text("Mensaje 'HOLA' recibido ✅")
    return mensaje_dot, texto

frames_totales = (len(coords) - 1) * 25 + 25
ani = animation.FuncAnimation(fig, update, frames=frames_totales, init_func=init,
                              blit=True, interval=150, repeat=False)

plt.show()

print("Imagen guardada como 'topologia_final_mejorada.png' en el directorio actual.")
