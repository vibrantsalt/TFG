import math
import matplotlib
import os
import pandas
import random
import shutil
import sys
import matplotlib.pyplot as plt


class Stroke(list):
    """Un trazo es una lista de coordenadas (x, y).

    Attributes:
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(self, initial_coordinate: tuple[int, int]):
        super().__init__([initial_coordinate])
        self.pd_predicted: int

    def append(self, coordinate: tuple[int, int]):
        super().append(coordinate)

    def set_pd_predicted(self, pd_predicted: int):
        self.pd_predicted = pd_predicted

    def get_x_coordinates_list(self) -> list[int]:
        """Devuelve la lista de coordenadas x del trazo."""

        return list(zip(*self))[0]

    def get_y_coordinates_list(self) -> list[int]:
        """Devuelve la lista de coordenadas y del trazo."""

        return list(zip(*self))[1]

    def distance(self, stroke: "Stroke") -> int:
        """Devuelve la distancia a otro trazo."""

        dist_a = math.dist((self[0][0], self[0][1]), (stroke[-1][0], stroke[-1][1]))
        dist_b = math.dist((self[-1][0], self[-1][1]), (stroke[0][0], stroke[0][1]))

        return round(min(dist_a, dist_b))

    def length(self) -> int:
        """Devuelve la longitud del trazo."""

        total_length = 0
        i = 1
        while i < len(self):
            total_length += math.dist(
                (self[i - 1][0], self[i - 1][1]), (self[i][0], self[i][1])
            )
            i += 1

        return round(total_length)


class Clipping:
    """Un recorte es una lista de trazos.

    Attributes:
        strokes_list: list. Trazos del recorte.
        center_coordinate: tuple. (x, y) en el centro del recorte.
        side_size: int. Tamaño del lado del recorte.
        jump_size: int. Coordenadas que se saltan entre recortes.
        letters_set: LetterSet. Conjunto de letras al que pertenece el recorte.
        number: int. Posición del recorte en el conjunto de letras.
        folder: str. Directorio en el que se encuentra el recorte.
        name: str. Nombre del recorte.
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(
        self,
        strokes_list: list[Stroke],
        center_coordinate: tuple[int, int],
        side_size: int,
        jump_size: int,
        letters_set: "LetterSet",
        number: int,
    ):
        self.strokes_list = strokes_list
        self.center_coordinate = center_coordinate
        self.side_size = side_size
        self.jump_size = jump_size
        self.letters_set = letters_set
        self.number = number
        self.folder = f"{letters_set.get_task_number()}_{side_size}_{jump_size}"
        self.name = f"{letters_set.get_subject_id()}_{letters_set.number}_{number}.png"
        self.pd_predicted: int
        self.plot()

    def plot(self):
        """Genera el plot del recorte y lo guarda."""

        # Se crea el directorio si es necesario.
        os.makedirs(os.path.join("generated", self.folder), exist_ok=True)

        # Se genera el plot y se guarda si no existe.
        if not os.path.exists(os.path.join("generated", self.folder, self.name)):
            plt.ioff()
            matplotlib.rcParams["savefig.pad_inches"] = 0
            px = 1 / plt.rcParams["figure.dpi"]
            fig = plt.figure(figsize=(self.side_size * px, self.side_size * px))

            ax = plt.axes((0, 0, 1, 1), frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.autoscale(tight=True)

            for stroke in self.strokes_list:
                stroke_x_list = stroke.get_x_coordinates_list()
                stroke_y_list = stroke.get_y_coordinates_list()
                plt.plot(stroke_x_list, stroke_y_list, color="black")

            # Necesario para que el centro del recorte sea el correcto.
            x_axis = (
                self.center_coordinate[0] - self.side_size / 2,
                self.center_coordinate[0] + self.side_size / 2,
            )
            y_axis = (
                self.center_coordinate[1] - self.side_size / 2,
                self.center_coordinate[1] + self.side_size / 2,
            )
            plt.plot(x_axis, y_axis, alpha=0)

            plt.savefig(os.path.join("generated", self.folder, self.name))
            plt.close()

    def copy_clipping(self, clipping_type: str):
        """Copia el recorte al directorio correspondiente."""

        clipping_origin = os.path.join("generated", self.folder, self.name)
        clipping_dest = os.path.join("generated", clipping_type, self.name)

        shutil.copyfile(clipping_origin, clipping_dest)


class LetterSet:
    """Un conjunto de letras es una lista de trazos.

    Attributes:
        strokes_list: list. Trazos del conjunto de letras.
        task: Task. Tarea a la que pertenece el conjunto de letras.
        number: int. Posición del conjunto de letras en la tarea.
        clippings_list: list. Recortes de la letra.
        coordinates_clippings_dict: dict. Recortes a los que pertenece cada coordenada.
        predicted_strokes_list: list. Trazos generados tras la predicción.
        unpredicted_coordinates_list: list. Coordenadas sin recorte.
        predicted_h_length: int. Longitud total de los trazos H en distancia.
        predicted_pd_length: int. Longitud total de los trazos PD en distancia.
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(self, strokes_list: list[Stroke], task: "Task", number: int):
        self.strokes_list = strokes_list
        self.task = task
        self.number = number
        self.clippings_list: list[Clipping] = []
        self.coordinates_clippings_dict: dict[tuple[int, int], list[Clipping]] = {}
        self.predicted_strokes_list: list[Stroke] = []
        self.unpredicted_coordinates_list: list[tuple[int, int]] = []
        self.predicted_h_length: int = 0
        self.predicted_pd_length: int = 0
        self.pd_predicted: int

    def get_subject_id(self) -> int:
        """Devuelve la ID del sujeto al que pertenece el conjunto de letras."""

        return self.task.subject_id

    def get_task_number(self) -> int:
        """Devuelve el número de la tarea a la que pertenece el conjunto de letras."""

        return self.task.task_number

    def generate_clippings(self, clipping_side_size: int, clipping_jump_size: int):
        """Genera los recortes del conjunto de letras."""

        self.clippings_list = []
        self.coordinates_clippings_dict = {}

        clipping_number = 0
        for set_stroke in self.strokes_list:
            coordinate_i = 0
            while coordinate_i < len(set_stroke):
                (
                    clipping_strokes_list,
                    clipping_coordinates_list,
                ) = self._find_clipping_strokes(
                    set_stroke[coordinate_i], clipping_side_size
                )

                clipping = Clipping(
                    clipping_strokes_list,
                    set_stroke[coordinate_i],
                    clipping_side_size,
                    clipping_jump_size,
                    self,
                    clipping_number,
                )
                self.clippings_list.append(clipping)
                for coordinate in clipping_coordinates_list:
                    if coordinate in self.coordinates_clippings_dict:
                        self.coordinates_clippings_dict[coordinate].append(clipping)
                    else:
                        self.coordinates_clippings_dict[coordinate] = [clipping]

                clipping_number += 1
                coordinate_i += clipping_jump_size

    def _find_clipping_strokes(
        self, center_coordinate: tuple[int, int], clipping_side_size
    ) -> tuple[list[Stroke], list[tuple[int, int]]]:
        """Devuelve la lista de los trazos del recorte."""

        clipping_strokes_list = []
        clipping_coordinates_list = []
        max_range = (clipping_side_size - 1) / 2
        from_in_range = False
        for set_stroke in self.strokes_list:
            current_stroke = set_stroke
            for current_coordinate in set_stroke:
                x_distance = abs(center_coordinate[0] - current_coordinate[0])
                y_distance = abs(center_coordinate[1] - current_coordinate[1])
                if (x_distance <= max_range) and (y_distance <= max_range):
                    clipping_coordinates_list.append(current_coordinate)
                    if from_in_range:
                        if current_stroke == set_stroke:
                            clipping_strokes_list[-1].append(current_coordinate)
                        else:
                            clipping_strokes_list.append(Stroke(current_coordinate))
                            current_stroke = set_stroke
                    else:
                        clipping_strokes_list.append(Stroke(current_coordinate))
                        from_in_range = True
                else:
                    from_in_range = False

        return clipping_strokes_list, clipping_coordinates_list

    def get_text_coordinates(self) -> tuple[int, int]:
        """Devuelve las coordenadas para escribir texto bajo el conjunto de letras."""

        height = sys.maxsize
        x_start = sys.maxsize
        x_end = 0

        for stroke in self.strokes_list:
            for coordinate in stroke:
                if coordinate[1] < height:
                    height = coordinate[1]
                if coordinate[0] < x_start:
                    x_start = coordinate[0]
                if coordinate[0] > x_end:
                    x_end = coordinate[0]

        x_coordinate = round((x_start + x_end) / 2 - 60)
        height -= 60

        return x_coordinate, height

    def generate_prediction_results(self):
        """Genera las predicciones a nivel de trazo y de conjunto de letras."""

        for stroke in self.strokes_list:
            current_predicted_pd = -1
            for coordinate in stroke:
                if coordinate in self.coordinates_clippings_dict:
                    coordinate_h = 0
                    coordinate_pd = 0
                    for clipping in self.coordinates_clippings_dict[coordinate]:
                        if clipping.pd_predicted == 0:
                            coordinate_h += 1
                        else:
                            coordinate_pd += 1
                    if coordinate_h == coordinate_pd:
                        predicted_pd = random.randint(0, 1)
                    elif coordinate_h > coordinate_pd:
                        predicted_pd = 0
                    else:
                        predicted_pd = 1

                    # Se generan los nuevos trazos en base
                    # a la predicción de las coordenadas.
                    if predicted_pd != current_predicted_pd:
                        new_stroke = Stroke(coordinate)
                        new_stroke.pd_predicted = predicted_pd
                        self.predicted_strokes_list.append(new_stroke)
                        current_predicted_pd = predicted_pd
                    else:
                        self.predicted_strokes_list[-1].append(coordinate)
                else:
                    self.unpredicted_coordinates_list.append(coordinate)
                    current_predicted_pd = -1

        for stroke in self.predicted_strokes_list:
            if stroke.pd_predicted == 0:
                self.predicted_h_length += stroke.length()
            else:
                self.predicted_pd_length += stroke.length()

        if self.predicted_h_length > self.predicted_pd_length:
            self.pd_predicted = 0
        else:
            self.pd_predicted = 1


class Task:
    """Una tarea es una lista de cinco conjuntos de letras.

    Attributes:
        subject_id: int. ID del sujeto al que pertenece la tarea.
        task_number: int. 2: "l" | 3: "le" | 4: "les"
        letters_sets_list: list. Lista de los conjuntos de letras que forman la tarea.
        predicted_h_length: int. Longitud total de los trazos H en distancia.
        predicted_pd_length: int. Longitud total de los trazos PD en distancia.
        pd_predicted: int. 0 para H y 1 para PD.
    """

    def __init__(self, subject_id: int, task_number: int, strokes_list: list[Stroke]):
        self.subject_id = subject_id
        self.task_number = task_number
        self.letters_sets_list = self._get_letters_sets(strokes_list)
        self.predicted_h_length: int = 0
        self.predicted_pd_length: int = 0
        self.pd_predicted: int

    def _get_letters_sets(self, strokes_list: list[Stroke]) -> list[LetterSet]:
        """Recibe una lista de trazos y los agrupa en varios conjuntos de letras."""

        # Los trazos se agrupan en torno a los cinco trazos más largos.
        sorted_strokes = sorted(strokes_list, reverse=True, key=len)
        first_strokes = sorted_strokes[:5]

        # Excepción: Tarea 2, Sujeto 26.
        if self.subject_id == 26 and self.task_number == 2:
            first_strokes = sorted_strokes[1:6]

        # Excepción: Tarea 3, Sujeto 54.
        if self.subject_id == 54 and self.task_number == 3:
            first_strokes = sorted_strokes[:3]
            first_strokes = first_strokes + sorted_strokes[4:6]

        # Excepción: Tarea 2, Sujetos (27, 57). Tarea 4, Sujetos (2, 48, 85).
        if (self.subject_id in (27, 57) and self.task_number == 2) or (
            self.subject_id in (2, 48, 85) and self.task_number == 4
        ):
            first_strokes = sorted_strokes[:6]

        # Se ordenan los trazos por su posición original.
        first_strokes = sorted(first_strokes, key=lambda stroke: stroke[0][0])

        sets_strokes_list = []
        for task_stroke in first_strokes:
            sets_strokes_list.append([task_stroke])

        for task_stroke in strokes_list:
            letters_set = -1
            distance = sys.maxsize
            stroke_exists = False
            for letters_set_i in range(len(sets_strokes_list)):
                for set_stroke in sets_strokes_list[letters_set_i]:
                    if task_stroke == set_stroke:
                        stroke_exists = True
                        break
                    current_distance = task_stroke.distance(set_stroke)
                    if current_distance < distance:
                        distance = current_distance
                        letters_set = letters_set_i
                if stroke_exists:
                    break
            if not stroke_exists:
                sets_strokes_list[letters_set].append(task_stroke)

        # Ahora se ordena cada lista de trazos por su posición original.
        for i in range(0, len(sets_strokes_list)):
            sets_strokes_list[i] = sorted(
                sets_strokes_list[i],
                key=lambda stroke: strokes_list.index(stroke),
            )

        # Por último se generan todos los conjuntos de letras.
        letters_sets_list = []
        letters_set_i = 0
        while letters_set_i < len(sets_strokes_list):
            letters_sets_list.append(
                LetterSet(sets_strokes_list[letters_set_i], self, letters_set_i)
            )
            letters_set_i += 1

        return letters_sets_list

    def generate_prediction_results(self):
        """Genera las predicciones a nivel de trazo,
        de conjunto de letras y de tarea."""

        for letters_sets in self.letters_sets_list:
            letters_sets.generate_prediction_results()
            self.predicted_h_length += letters_sets.predicted_h_length
            self.predicted_pd_length += letters_sets.predicted_pd_length

        if self.predicted_h_length > self.predicted_pd_length:
            self.pd_predicted = 0
        else:
            self.pd_predicted = 1


def load() -> tuple[dict[int, tuple[int, int]], dict[int, dict[int, Task]]]:
    """Devuelve dos diccionarios, ambos con la ID del sujeto como clave. En el primero
    el valor es una tupla con el estado PD y los años PD y en el otro una lista con las
    tareas 2, 3 y 4 del sujeto.
    """

    pahaw_file_path = os.path.join("PaHaW", "PaHaW_files", "corpus_PaHaW.xlsx")
    pahaw_data_frame = pandas.read_excel(pahaw_file_path)
    task_file_path_start = os.path.join("PaHaW", "PaHaW_public")
    task_file_path_end = "_1.svc"
    subjects_id_list = list(map(int, pahaw_data_frame["ID"].to_list()))
    subjects_pd_status_list = [
        0 if e == "H" else 1 for e in pahaw_data_frame["Disease"].to_list()
    ]
    subject_pd_years_list = list(
        map(int, pahaw_data_frame["Length of PD"].fillna(0).to_list())
    )

    subjects_pd_status_years_dict = {}
    subjects_tasks_dict = {}
    subject_i = 0
    while subject_i < len(subjects_id_list):
        subject_id = subjects_id_list[subject_i]
        pd_status_years = (
            subjects_pd_status_list[subject_i],
            subject_pd_years_list[subject_i],
        )
        for task_number in range(2, 5):
            task_file_path_mid = os.path.join(
                f"{subject_id:05d}", f"{subject_id:05d}__{task_number}"
            )
            task_file_path = os.path.join(
                task_file_path_start, task_file_path_mid + task_file_path_end
            )
            task_strokes_list = []
            with open(task_file_path, encoding="utf-8") as task_file:
                # Se salta la primera línea.
                task_file.readline()
                from_on_air = True

                while True:
                    line = task_file.readline()
                    if not line:
                        break
                    # Si la coordenada está sobre el papel.
                    if line.split()[3] == "1":
                        coordinate = int(line.split()[1]), int(line.split()[0])
                        if from_on_air:
                            task_strokes_list.append(Stroke(coordinate))
                            from_on_air = False
                        else:
                            task_strokes_list[-1].append(coordinate)
                    else:
                        from_on_air = True

            subjects_pd_status_years_dict[subject_id] = pd_status_years
            if task_number == 2:
                subjects_tasks_dict[subject_id] = {
                    task_number: Task(subject_id, task_number, task_strokes_list)
                }
            else:
                subjects_tasks_dict[subject_id][task_number] = Task(
                    subject_id, task_number, task_strokes_list
                )
        subject_i += 1

    return subjects_pd_status_years_dict, subjects_tasks_dict
