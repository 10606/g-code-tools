#!/usr/bin/env python3
"""
text2gcode.py — превращает текст в G-code (контуры букв).

Зависимости:  pip install matplotlib shapely   (shapely нужен только для --tool-dia > 0)

Примеры:
  python text2gcode.py "Привет, мир!" -o hello.nc
  python text2gcode.py "Hello" --height 10 --mode pen --font C:/Windows/Fonts/arial.ttf
  python text2gcode.py "Лазер" --mode laser --power 800 --feed 1200
  python text2gcode.py "Фреза" --mode router --depth -0.5 --passes 2 --feed 300
  python text2gcode.py "Фреза" --tool-dia 3 --comp outside   (компенсация диаметра фрезы)
  python text2gcode.py "Фреза" --tool-dia 1 --comp inside --embolden 0.06   (чуть утолщить штрихи)
  python text2gcode.py -i text.txt -o out.nc      (текст из файла, многострочный)

Режимы:
  router — фреза/гравёр: Z поднимается до --safe-z и опускается до --depth
  pen    — плоттер: Z=--safe-z (перо поднято), Z=--depth (перо опущено)
  laser  — лазер: без Z, включение M3 S<power> / выключение M5
"""
import argparse
import math
import os
import sys
import traceback

from matplotlib.font_manager import FontProperties, findfont
from matplotlib.textpath import TextPath


DEBUG = False


def dbg(msg):
    """Отладочное сообщение (включается флагом --debug), выводится в stderr."""
    if DEBUG:
        print(f"[debug] {msg}", file=sys.stderr)


def build_paths(text, font_path, height, line_spacing, align):
    """Возвращает список контуров [[(x, y), ...], ...] в мм. Начало координат — левый нижний угол."""
    fp = FontProperties(fname=font_path) if font_path else FontProperties(family="DejaVu Sans")
    # Размер шрифта подбираем так, чтобы высота заглавной буквы была ~ height
    ref = TextPath((0, 0), "H", size=100, prop=fp).get_extents().height
    size = 100 * height / ref if ref else height

    lines = text.split("\n")
    line_paths = []
    widths = []
    for line in lines:
        tp = TextPath((0, 0), line, size=size, prop=fp)
        polys = tp.to_polygons(closed_only=False)  # кривые Безье уже разбиты на отрезки
        width = tp.get_extents().x1 if line.strip() and len(tp.vertices) else 0
        line_paths.append(polys)
        widths.append(width)

    max_w = max(widths) if widths else 0
    step = height * line_spacing
    result = []
    n = len(lines)
    for i, polys in enumerate(line_paths):
        dx = {"left": 0, "center": (max_w - widths[i]) / 2, "right": max_w - widths[i]}[align]
        dy = (n - 1 - i) * step
        for poly in polys:
            if len(poly) >= 2:
                result.append([(x + dx, y + dy) for x, y in poly])
    return result


def _densify(coords, step):
    """Добавляет точки на границу, чтобы расстояние между ними было не больше step."""
    out = []
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        for i in range(n):
            t = i / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def _axis_paths(edges, r, log=None):
    """edges: [(a, b, wa, wb)] — рёбра средней оси буквы (a, b — точки (x, y);
    wa, wb — расстояние от них до границы). Возвращает полилинии по участкам оси,
    где штрих уже диаметра фрезы (w < r)."""
    say = log or (lambda m: None)
    WMIN = 0.12 * r      # ниже — рёбра, лежащие у самой границы (иголки Вороного)
    RATIO = 0.7          # ответвление считаем шумом, если w на его конце < RATIO * w в развилке
    MIN_LEN = 0.3 * r    # минимальная длина тонкого участка
    TOL = 1.03           # допуск: w чуть выше r тоже считаем "тонким" (склеивает мелкие разрывы)

    def key(p):
        return (round(p[0], 5), round(p[1], 5))

    E, pos, nw = [], {}, {}
    for a, b, wa, wb in edges:
        if min(wa, wb) < WMIN:
            continue
        ka, kb = key(a), key(b)
        if ka == kb:
            continue
        pos[ka], pos[kb] = a, b
        nw[ka], nw[kb] = wa, wb
        E.append((ka, kb, math.hypot(a[0] - b[0], a[1] - b[1])))
    say(f"рёбер оси после отсечения границы (w >= {WMIN:.2f} мм): {len(E)} из {len(edges)}")
    if not E:
        return []

    node_edges = {}
    for i, (ka, kb, _) in enumerate(E):
        node_edges.setdefault(ka, []).append(i)
        node_edges.setdefault(kb, []).append(i)

    def other(e, n):
        return E[e][1] if E[e][0] == n else E[e][0]

    # 1) убираем висячие ответвления-иголки (итеративно)
    alive = set(range(len(E)))
    pruned = 0
    deg = {}
    for i in alive:
        for n in E[i][:2]:
            deg[n] = deg.get(n, 0) + 1
    cut = set()
    # все висячие ответвления оцениваются по степеням узлов на начало прохода,
    # чтобы обе вилки на конце полосы отбрасывались вместе
    for n0, dg in deg.items():
        if dg != 1:
            continue
        cur, prev, leg, length = n0, None, [], 0.0
        while True:
            e = next(i for i in node_edges[cur] if i in alive and i != prev)
            leg.append(e)
            length += E[e][2]
            nxt = other(e, cur)
            if deg[nxt] != 2:
                break
            cur, prev = nxt, e
        if deg[nxt] >= 3:
            wj = nw[nxt]
            if nw[n0] < RATIO * wj or length < 0.5 * wj:
                cut.update(leg)
    alive -= cut
    pruned += len(cut)
    say(f"убрано рёбер-ответвлений (шум Вороного, вилки, спуры к углам): {pruned}; осталось {len(alive)}")

    # 2) тонкие рёбра: ширина < диаметра (w < r)
    thin = {i for i in alive if min(nw[E[i][0]], nw[E[i][1]]) < r * TOL}
    say(f"из них тонких (w < {r * TOL:.3f} мм): {len(thin)}")

    # 3) компоненты связности тонких рёбер
    seen, comps = set(), []
    for i in thin:
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            j = stack.pop()
            if j in comp:
                continue
            comp.add(j)
            for n in E[j][:2]:
                stack.extend(k for k in node_edges[n] if k in thin)
        seen |= comp
        comps.append(comp)

    lines = []
    for ci, es in enumerate(comps):
        length = sum(E[i][2] for i in es)
        ws = [min(nw[E[i][0]], nw[E[i][1]]) for i in es]
        ok = length >= MIN_LEN
        say(f"  тонкий участок {ci}: рёбер {len(es)}, длина {length:.2f} мм, "
            f"w {min(ws):.2f}..{max(ws):.2f} мм -> {'ОСТАВЛЕН' if ok else 'отброшен (слишком короткий)'}")
        if not ok:
            continue
        deg = {}
        for i in es:
            for n in E[i][:2]:
                deg[n] = deg.get(n, 0) + 1
        used = set()
        starts = [n for n, dg in deg.items() if dg != 2] or [next(iter(deg))]
        for s0 in starts:
            for e0 in [i for i in node_edges[s0] if i in es and i not in used]:
                line, cur, e = [pos[s0]], s0, e0
                while True:
                    used.add(e)
                    nxt = other(e, cur)
                    line.append(pos[nxt])
                    if deg[nxt] != 2:
                        break
                    more = [i for i in node_edges[nxt] if i in es and i not in used]
                    if not more:
                        break
                    cur, e = nxt, more[0]
                lines.append(line)
    return lines


def _axis_edges(geom, radius):
    """Рёбра средней оси буквы: [(a, b, wa, wb)], где wa, wb — расстояние от концов до границы."""
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import voronoi_diagram
    from shapely.prepared import prep

    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    step = max(radius / 6.0, 0.03)
    pts = []
    for p in polys:
        pts += _densify(list(p.exterior.coords), step)
        for h in p.interiors:
            pts += _densify(list(h.coords), step)
    pts = list({(round(x, 6), round(y, 6)) for x, y in pts})
    dbg(f"ось: шаг {step:.3f} мм, точек границы {len(pts)}")
    if len(pts) < 4:
        return []

    inside = prep(geom)
    boundary = geom.boundary
    raw = list(voronoi_diagram(MultiPoint(pts), edges=True).geoms)
    edges = []
    for e in raw:
        if e.is_empty:
            continue
        for e_i in e.geoms:
            if e_i.is_empty or not inside.contains(e_i):
                continue
            c = list(e_i.coords)
            dbg(f"ось: ребро Вороного (кандидат) {c}")
            for a, b in zip(c[:-1], c[1:]):
                edges.append((a, b, boundary.distance(Point(a)), boundary.distance(Point(b))))
    dbg(f"ось: рёбер Вороного всего {len(raw)}, внутри буквы {len(edges)}")
    return edges


def apply_tool_offset(paths, radius, side, centerline=True, embolden=0.0):
    """Смещает контуры на радиус инструмента (эквидистанта, скругление углов).

    side='outside' — центр фрезы идёт СНАРУЖИ буквы: сама буква получается ровно
                     номинального размера (вырезание букв, обход буквы).
    side='inside'  — центр фрезы идёт ВНУТРИ буквы: внешний край канавки лежит
                     ровно по номинальному контуру (буква-углубление).
                     Участки уже диаметра фрезы (фреза не проходит) при
                     centerline=True проходятся по средней линии.
    embolden       — синтетическая "толщина шрифта": контур буквы расширяется на
                     embolden мм с каждой стороны (<0 — сужается) до расчёта траектории.
                     Если штрихи чуть уже фрезы, небольшого утолщения хватает, чтобы
                     фреза проходила везде и траектория не рвалась.
    """
    try:
        import shapely
        from shapely.geometry import Polygon
    except ImportError:
        sys.exit("Для --tool-dia > 0 нужен shapely:  pip install shapely")

    dbg(f"shapely {getattr(shapely, '__version__', '?')}; радиус {radius} мм; сторона {side}; "
        f"поиск тонких участков: {'вкл' if centerline else 'выкл'}")

    # Собираем букву по правилу чётности: вложенный контур = отверстие (о, а, В, ...)
    geom = Polygon()
    for ring in paths:
        if len(ring) < 3:
            continue
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        geom = geom.symmetric_difference(poly)

    if embolden:
        geom = geom.buffer(embolden, 16)
        dbg(f"утолщение контура на {embolden:+.3f} мм с каждой стороны")

    b = geom.bounds if not geom.is_empty else None
    dbg(f"геометрия: тип {geom.geom_type}, площадь {geom.area:.2f} мм2, валидна {geom.is_valid}, "
        f"границы {tuple(round(v, 2) for v in b) if b else None}, "
        f"частей {len(geom.geoms) if hasattr(geom, 'geoms') else 1}")

    d = radius if side == "outside" else -radius
    off = geom.buffer(d, 16)
    dbg(f"смещённый контур: пуст {off.is_empty}, площадь {off.area:.3f} мм2, "
        f"частей {len(off.geoms) if hasattr(off, 'geoms') else (0 if off.is_empty else 1)}")

    # Тонкие участки: где штрих уже диаметра фрезы смещённый контур рвётся ->
    # ведём центр фрезы по средней линии буквы.
    extra = []
    if side != "inside":
        dbg("поиск тонких участков пропущен: он работает только для --comp inside")
    elif not centerline:
        dbg("поиск тонких участков пропущен: указан --no-centerline")
    else:
        try:
            from shapely.geometry import LineString
            edges = _axis_edges(geom, radius)
            tol = max(radius / 6.0, 0.03) * 0.2
            for ln in _axis_paths(edges, radius, dbg):
                extra.append([(x, y) for x, y in LineString(ln).simplify(tol).coords])
            dbg(f"итого линий по средней оси: {len(extra)}")
        except Exception as ex:  # не ломаем генерацию из-за геометрии
            print(f"ВНИМАНИЕ: не удалось построить средние линии тонких участков: {ex!r}",
                  file=sys.stderr)
            if DEBUG:
                traceback.print_exc()
            extra = []

    if off.is_empty and not extra:
        sys.exit("После смещения ничего не осталось: штрихи букв тоньше диаметра фрезы. "
                 "Увеличьте --height или возьмите фрезу меньшего диаметра.")

    polys = [] if off.is_empty else (list(off.geoms) if hasattr(off, "geoms") else [off])
    result = []
    for p in polys:
        result.append(list(p.exterior.coords))
        for hole in p.interiors:
            result.append(list(hole.coords))

    if extra:
        print(f"Тонких участков (уже диаметра фрезы): {len(extra)} линий пойдут по средней линии")
        result += extra
    elif side == "inside" and centerline:
        print("Тонких участков не найдено (для подробностей запустите с --debug)")
    elif side == "inside" and not centerline:
        print("ВНИМАНИЕ: тонкие участки уже диаметра фрезы пропущены (--no-centerline).",
              file=sys.stderr)
    return result


def generate_gcode(paths, a):
    g = []
    add = g.append
    add("; Generated by text2gcode.py")
    add(f"; Text height: {a.height} mm, mode: {a.mode}, tool dia: {a.tool_dia} mm ({a.comp if a.tool_dia > 0 else 'no comp'}), embolden: {a.embolden} mm")
    add("; F (скорость) указана явно в КАЖДОЙ команде движения, чтобы остановка/возврат на станке её не сбивали")
    add("G21 ; units: millimeters")
    add("G90 ; absolute coordinates")
    add("G94 ; feed rate in mm/min")
    add("G54 ; work coordinate system")
    add("G17 ; XY plane")

    y_base = -a.height / 2.
    xs = [x for p in paths for x, _ in p]
    x_base = -(max(xs) + min(xs)) / 2.
    
    angle = a.angle * math.pi / 180.;
    m_x_x = math.cos(angle);
    m_x_y = -math.sin(angle);
    m_y_x = math.sin(angle);
    m_y_y = math.cos(angle);
    
    ox, oy = a.x0, a.y0
    # Подъём делаем рабочей подачей (G1), а не ускоренно (G0): на быстром подъёме
    # фреза может выдирать волокна древесины.
    retract = a.retract_feed if a.retract_feed else a.feed
    rapid = a.rapid_feed

    def mv(cmd, args, feed):
        """Любая команда движения (G0/G1/G2/G3) всегда пишется с явным F."""
        return f"{cmd} {args} F{feed}"

    def pt(p):
        x_from_center = p[0] + x_base;
        y_from_center = p[1] + y_base;
        x_rotate = x_from_center * m_x_x + y_from_center * m_x_y + a.x0;
        y_rotate = x_from_center * m_y_x + y_from_center * m_y_y + a.y0;
        return f"X{x_rotate:.3f} Y{y_rotate:.3f}"

    if a.mode == "laser":
        add("M5")
        for p in paths:
            add(mv("G0", pt(p[0]), rapid))
            add(f"M3 S{a.power}")
            for q in p[1:]:
                add(mv("G1", pt(q), a.feed))
            add("M5")
    else:
        add(mv("G0", f"Z{a.safe_z}", rapid))
        if a.mode == "router" and a.spindle:
            add(f"M3 S{a.spindle}")
        for k in range(1, a.passes + 1):
            z = a.depth * k / a.passes
            add(f"; pass {k}/{a.passes}")
            for p in paths:
                add(mv("G0", pt(p[0]), rapid))
                add(mv("G1", f"Z{z:.3f}", a.plunge))
                for q in p[1:]:
                    add(mv("G1", pt(q), a.feed))
                add(mv("G1", f"Z{a.safe_z}", retract))
        if a.mode == "router" and a.spindle:
            add("M5")

    # add(mv("G0", "X0 Y0", rapid))
    add("M2")
    return "\n".join(g) + "\n"


def append_gcode(path, code):
    """Дописывает code в конец существующего файла, не трогая остальное содержимое.

    Единственное исключение: если файл заканчивается командой конца программы (M2/M30),
    эта последняя строка удаляется — иначе дописанный код после неё не выполнился бы.
    Возвращает удалённую команду (или None).
    """
    removed = None
    data = b""
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
    if data.strip():
        body = data.rstrip()
        nl = body.rfind(b"\n")
        cmd = body[nl + 1:].split(b";")[0].strip().upper()
        if cmd in (b"M2", b"M30"):
            removed = cmd.decode()
            with open(path, "rb+") as f:
                f.truncate(nl + 1 if nl >= 0 else 0)
            data = body[:nl + 1] if nl >= 0 else b""
    out = code
    if b"\r\n" in data:                      # сохраняем стиль переводов строк файла
        out = out.replace("\n", "\r\n")
    with open(path, "ab") as f:
        if data.strip():
            if not data.endswith(b"\n"):
                f.write(b"\r\n" if b"\r\n" in data else b"\n")
            f.write(b"\r\n" if b"\r\n" in data else b"\n")      # пустая строка-разделитель
        f.write(out.encode("utf-8"))
    return removed


def main():
    ap = argparse.ArgumentParser(description="Текст → G-code")
    ap.add_argument("text", nargs="?", help="Текст (\\n — перенос строки)")
    ap.add_argument("-i", "--input", help="Прочитать текст из файла")
    ap.add_argument("-o", "--output", default="output.nc", help="Выходной файл")
    ap.add_argument("--append", action="store_true",
                    help="Не перезаписывать выходной файл, а дописать в его конец "
                         "(завершающий M2/M30 существующего файла при этом удаляется)")
    ap.add_argument("--font", help="Путь к TTF/OTF (по умолчанию DejaVu Sans)")
    ap.add_argument("--height", type=float, default=10.0, help="Высота заглавных букв, мм")
    ap.add_argument("--angle", type=float, default=0.0, help="Угол поворота, градусов")
    ap.add_argument("--line-spacing", type=float, default=1.4, help="Межстрочный интервал (в высотах)")
    ap.add_argument("--align", choices=["left", "center", "right"], default="left")
    ap.add_argument("--mode", choices=["router", "pen", "laser"], default="router")
    ap.add_argument("--feed", type=int, default=500, help="Подача при резке, мм/мин")
    ap.add_argument("--plunge", type=int, default=150, help="Подача при опускании по Z")
    ap.add_argument("--safe-z", type=float, default=3.0, help="Безопасная высота Z")
    ap.add_argument("--depth", type=float, default=-0.3, help="Глубина Z (для pen — высота опущенного пера)")
    ap.add_argument("--passes", type=int, default=1, help="Число проходов по глубине")
    ap.add_argument("--spindle", type=int, default=0, help="Обороты шпинделя (0 — не включать)")
    ap.add_argument("--power", type=int, default=1000, help="Мощность лазера S")
    ap.add_argument("--tool-dia", type=float, default=0.0,
                    help="Диаметр фрезы/луча/пера, мм (0 — без компенсации)")
    ap.add_argument("--comp", choices=["outside", "inside"], default="outside",
                    help="Куда смещать центр фрезы: outside — снаружи буквы, inside — внутри")
    ap.add_argument("--embolden", type=float, default=0.0,
                    help="Утолщить контур букв на N мм с каждой стороны (<0 — тоньше); "
                         "действует вместе с --tool-dia. Аналог выбора более жирного шрифта")
    ap.add_argument("--rapid-feed", type=int, default=1500,
                    help="Значение F для холостых ходов G0, мм/мин (GRBL его игнорирует, "
                         "на Marlin и подобных оно задаёт скорость холостых ходов)")
    ap.add_argument("--retract-feed", type=int, default=None,
                    help="Подача при подъёме фрезы из материала, мм/мин (по умолчанию = --feed)")
    ap.add_argument("--no-centerline", action="store_true",
                    help="При --comp inside НЕ проходить тонкие участки по средней линии")
    ap.add_argument("--debug", action="store_true",
                    help="Подробные отладочные сообщения (в stderr) о поиске тонких участков")
    ap.add_argument("--x0", type=float, default=0.0, help="Смещение по X")
    ap.add_argument("--y0", type=float, default=0.0, help="Смещение по Y")
    a = ap.parse_args()
    global DEBUG
    DEBUG = a.debug

    if a.input:
        with open(a.input, encoding="utf-8") as f:
            text = f.read().rstrip("\n")
    elif a.text:
        text = a.text.replace("\\n", "\n")
    else:
        ap.error("Нужен текст или --input")

    paths = build_paths(text, a.font, a.height, a.line_spacing, a.align)
    if not paths:
        sys.exit("Не удалось построить контуры (проверьте шрифт и текст)")

    if a.embolden and a.tool_dia <= 0:
        print("ВНИМАНИЕ: --embolden действует только вместе с --tool-dia > 0, параметр проигнорирован.",
              file=sys.stderr)
    if a.tool_dia > 0:
        def bbox(ps):
            xs = [x for p in ps for x, _ in p]
            ys = [y for p in ps for _, y in p]
            return min(xs), min(ys), max(xs), max(ys)
        before = bbox(paths)
        n_before = len(paths)
        paths = apply_tool_offset(paths, a.tool_dia / 2.0, a.comp, not a.no_centerline, a.embolden)
        after = bbox(paths)
        print(f"Диаметр {a.tool_dia} мм -> радиус смещения {a.tool_dia / 2.0} мм ({a.comp})")
        print(f"Границы буквы  X {before[0]:.2f}..{before[2]:.2f}  Y {before[1]:.2f}..{before[3]:.2f}")
        print(f"Путь центра    X {after[0]:.2f}..{after[2]:.2f}  Y {after[1]:.2f}..{after[3]:.2f}")
        print(f"Контуров: было {n_before}, стало {len(paths)}")

    code = generate_gcode(paths, a)
    if a.append:
        existed = os.path.exists(a.output) and os.path.getsize(a.output) > 0
        removed = append_gcode(a.output, code)
        if removed:
            print(f"Из конца существующего файла удалена команда {removed} (иначе дописанный код не выполнился бы)")
        print(f"{'Дописано в конец' if existed else 'Файл не существовал или был пуст, создан'}: {a.output}")
    else:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Готово: {a.output}")

    xs = [x for p in paths for x, _ in p]
    ys = [y for p in paths for _, y in p]
    print(f"Контуров: {len(paths)}, размер: {max(xs)-min(xs):.1f} x {max(ys)-min(ys):.1f} мм")


if __name__ == "__main__":
    main()
