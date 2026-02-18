import cv2
import mediapipe as mp
import numpy as np
import random

class InterfazVisual:
    def __init__(self):
        self.imagenes = {
            "PIEDRA": cv2.resize(cv2.imread("piedra.png", cv2.IMREAD_UNCHANGED), (150, 150)),
            "PAPEL": cv2.resize(cv2.imread("papel.png", cv2.IMREAD_UNCHANGED), (150, 150)),
            "TIJERA": cv2.resize(cv2.imread("tijera.png", cv2.IMREAD_UNCHANGED), (150, 150))
        }

    def dibujar_rectangulo_transparente(self, imagen, punto1, punto2, color, opacidad=0.5):
        sub_imagen = imagen[punto1[1]:punto2[1], punto1[0]:punto2[0]]
        rectangulo_color = np.full(sub_imagen.shape, color, dtype=np.uint8)

        resultado = cv2.addWeighted(sub_imagen, 1 - opacidad, rectangulo_color, opacidad, 0)
        imagen[punto1[1]:punto2[1], punto1[0]:punto2[0]] = resultado
        return imagen

    def superponer_imagen(self, fondo, jugada, x, y):
        if jugada not in self.imagenes or self.imagenes[jugada] is None:
            return fondo

        img_overlay = self.imagenes[jugada]
        alto, ancho = img_overlay.shape[:2]

        if y + alto > fondo.shape[0] or x + ancho > fondo.shape[1]:
            return fondo

        if img_overlay.shape[2] == 4:
            alpha = img_overlay[:, :, 3] / 255.0
            for c in range(0, 3):
                fondo[y:y + alto, x:x + ancho, c] = (alpha * img_overlay[:, :, c] +
                                                     (1.0 - alpha) * fondo[y:y + alto, x:x + ancho, c])
        else:
            fondo[y:y + alto, x:x + ancho] = img_overlay[:, :, :3]

        return fondo


class CerebroIA:
    def __init__(self, tamano_memoria=15):
        self.ultima_jugada_usuario = None
        self.memoria_transiciones = {
            "PIEDRA": {"PIEDRA": 0, "PAPEL": 0, "TIJERA": 0},
            "PAPEL": {"PIEDRA": 0, "PAPEL": 0, "TIJERA": 0},
            "TIJERA": {"PIEDRA": 0, "PAPEL": 0, "TIJERA": 0}
        }
        self.cola_transiciones = []
        self.limite_memoria = tamano_memoria
        self.opciones = ["PIEDRA", "PAPEL", "TIJERA"]

    def registrar_jugada(self, jugada_actual):
        if jugada_actual in ["NADA", "DESCONOCIDO"]:
            return

        if self.ultima_jugada_usuario:
            transicion = (self.ultima_jugada_usuario, jugada_actual)

            self.memoria_transiciones[transicion[0]][transicion[1]] += 1
            self.cola_transiciones.append(transicion)

            if len(self.cola_transiciones) > self.limite_memoria:
                vieja_transicion = self.cola_transiciones.pop(0)
                self.memoria_transiciones[vieja_transicion[0]][vieja_transicion[1]] -= 1

        self.ultima_jugada_usuario = jugada_actual

    def generar_jugada(self):
        if not self.ultima_jugada_usuario:
            return random.choice(self.opciones)

        historial = self.memoria_transiciones[self.ultima_jugada_usuario]

        if sum(historial.values()) == 0:
            return random.choice(self.opciones)

        prediccion_usuario = max(historial, key=historial.get)

        ganar_a = {"PIEDRA": "PAPEL", "PAPEL": "TIJERA", "TIJERA": "PIEDRA"}

        print(f"memoria activa: {len(self.cola_transiciones)} movimientos")
        return ganar_a[prediccion_usuario]


class MotorJuego:
    def __init__(self):
        self.puntos_jugador = 0
        self.puntos_ia = 0
        self.reglas = {"PIEDRA": "TIJERA", "PAPEL": "PIEDRA", "TIJERA": "PAPEL"}

    def obtener_resultado(self, jugada_u, jugada_ia):
        if jugada_u == jugada_ia: return "EMPATE"

        if self.reglas[jugada_u] == jugada_ia:
            self.puntos_jugador += 1
            return "GANASTE"

        self.puntos_ia += 1
        return "PERDISTE"


class DetectorManos:
    def __init__(self):
        self.mp_manos = mp.solutions.hands
        self.detector = self.mp_manos.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_dibujo = mp.solutions.drawing_utils

    def encontrar_jugada(self, imagen_rgb):
        resultados = self.detector.process(imagen_rgb)
        puntos_mano = None
        jugada = "NADA"

        if resultados.multi_hand_landmarks:
            puntos_mano = resultados.multi_hand_landmarks[0]
            jugada = self._clasificar_gesto(puntos_mano)

        return resultados, puntos_mano, jugada

    def _clasificar_gesto(self, puntos):
        dedos = []
        for p, b in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            dedos.append(puntos.landmark[p].y < puntos.landmark[b].y)

        pulgar = puntos.landmark[4].y < puntos.landmark[3].y
        total = dedos.count(True) + (1 if pulgar else 0)

        if total <= 1: return "PIEDRA"
        if total >= 4: return "PAPEL"
        if total == 2 and dedos[0] and dedos[1]: return "TIJERA"
        return "DESCONOCIDO"


def ejecutar_juego():
    cap = cv2.VideoCapture(0)
    ia = CerebroIA()
    motor = MotorJuego()
    vision = DetectorManos()

    estado = "ESPERANDO"  #ESPERANDO, CONTANDO, RESULTADO
    temporizador = 0
    jugada_ia = ""
    res_texto = ""
    jugada_usuario_final = ""

    visual = InterfazVisual()

    cv2.namedWindow('Agente Piedra Papel Tijera', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('Agente Piedra Papel Tijera', 900, 600)

    while cap.isOpened():
        exito, imagen = cap.read()
        if not exito: break

        imagen = cv2.flip(imagen, 1)
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        resultados, puntos, jugada_actual = vision.encontrar_jugada(imagen_rgb)

        if puntos:
            vision.mp_dibujo.draw_landmarks(imagen, puntos, vision.mp_manos.HAND_CONNECTIONS)

        tecla = cv2.waitKey(1) & 0xFF

        visual.dibujar_rectangulo_transparente(imagen, (0, 0), (640, 60), (0, 0, 0), opacidad=0.6)

        cv2.putText(imagen, f"Jugador: {motor.puntos_jugador} | IA: {motor.puntos_ia}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if estado == "ESPERANDO":
            visual.dibujar_rectangulo_transparente(imagen, (70, 410), (540, 470), (50, 50, 50), opacidad=0.7)
            cv2.putText(imagen, "Presiona 'Enter' para jugar", (90, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if tecla == 13: #13 es enter en ascii
                estado = "CONTANDO"
                temporizador = 30
                jugada_ia = ia.generar_jugada()

        elif estado == "CONTANDO":
            temporizador -= 1
            conteo = int(temporizador / 10) + 1
            cv2.putText(imagen, str(conteo), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)

            if temporizador <= 0:
                jugada_usuario_final = jugada_actual
                if jugada_usuario_final != "DESCONOCIDO" and jugada_usuario_final != "NADA":
                    res_texto = motor.obtener_resultado(jugada_usuario_final, jugada_ia)
                    ia.registrar_jugada(jugada_usuario_final)
                    estado = "RESULTADO"
                    temporizador = 40
                else:
                    estado = "ESPERANDO"

        elif estado == "RESULTADO":
            imagen = visual.superponer_imagen(imagen, jugada_ia, 450, 100)

            imagen = visual.superponer_imagen(imagen, jugada_usuario_final, 50, 100)

            cv2.putText(imagen, res_texto, (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

            temporizador -= 1
            if temporizador <= 0: estado = "ESPERANDO"

        cv2.imshow('Agente Piedra Papel Tijera', imagen)
        if tecla == 27: break #27 es esc en ascii

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ejecutar_juego()