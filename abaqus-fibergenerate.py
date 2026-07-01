# -*- coding: utf-8 -*-
# Abaqus Python 2.7 script
# Read fiber centers from Excel and create fiber cylinders one by one.
#
# Recommended unit system:
# length: mm
# stress: MPa
# force : N

from abaqus import *
from abaqusConstants import *
import regionToolset
import xlrd
import os


# ============================================================
# 1. User settings
# ============================================================

# 修改成你的坐标文件路径
# 注意：这里应该是含有 for_abaqus 工作表的坐标文件
EXCEL_FILE = r"E:\temp\80%\fiber_centers.xls"

# 周期性建模读取 for_abaqus
SHEET_NAME = "for_abaqus"

# True = 读取原始纤维 + 周期复制纤维
INCLUDE_PERIODIC_COPIES = True

# RVE size in Abaqus unit: mm
# 50 um = 0.05 mm
# 5  um = 0.005 mm
Lx = 0.05
Ly = 0.05
Lz = 0.005

MODEL_NAME = "RVE_Model"

# 第一次运行建议 False，先确认所有纤维能正常生成
# 如果所有单根纤维都能生成，再改成 True 尝试合并
MERGE_FIBERS = False

# 创建临时纤维材料和截面
CREATE_DUMMY_SECTION = True

# 保存 cae 文件
SAVE_CAE = False
CAE_FILE = r"E:\temp\80%\rve_fibers.cae"


# ============================================================
# 2. Helper functions
# ============================================================

def str_clean(x):
    try:
        return str(x).strip()
    except:
        try:
            return x.encode("utf-8").strip()
        except:
            return str(x).strip()


def find_col(header, name):
    name = name.lower()
    for i, h in enumerate(header):
        if str_clean(h).lower() == name:
            return i
    raise ValueError("Cannot find column: %s" % name)


def is_true_value(v):
    """
    For Excel boolean / string boolean.
    """
    if isinstance(v, bool):
        return v

    s = str(v).strip().lower()

    if s in ["true", "1", "1.0", "yes", "y"]:
        return True
    else:
        return False


def get_model(model_name):
    if model_name in mdb.models.keys():
        return mdb.models[model_name]
    else:
        return mdb.Model(name=model_name)


def delete_old_fibers(model):
    """
    删除之前生成的 Fiber_* 实例和 Part，避免重复运行时混乱。
    """
    a = model.rootAssembly

    # 删除 Assembly 中旧实例
    old_instances = list(a.instances.keys())
    for iname in old_instances:
        if iname.startswith("Fiber_") or iname.startswith("Fibers"):
            try:
                a.deleteFeatures(featureNames=(iname,))
            except:
                try:
                    del a.instances[iname]
                except:
                    pass

    # 删除旧 Part
    old_parts = list(model.parts.keys())
    for pname in old_parts:
        if pname.startswith("Fiber_") or pname.startswith("Fibers"):
            try:
                del model.parts[pname]
            except:
                pass


def read_fiber_data_from_excel(excel_file, sheet_name, include_periodic_copies=True):
    """
    Required columns:
    x_mm, y_mm, r_mm

    Optional columns:
    id, parent_id, is_periodic_copy
    """
    if not os.path.exists(excel_file):
        raise IOError("Excel file does not exist: %s" % excel_file)

    book = xlrd.open_workbook(excel_file)

    if sheet_name not in book.sheet_names():
        raise ValueError(
            "Sheet '%s' not found. Available sheets: %s"
            % (sheet_name, book.sheet_names())
        )

    sheet = book.sheet_by_name(sheet_name)

    header = []
    for c in range(sheet.ncols):
        header.append(str_clean(sheet.cell_value(0, c)))

    lower_header = [h.lower() for h in header]

    col_x = find_col(header, "x_mm")
    col_y = find_col(header, "y_mm")
    col_r = find_col(header, "r_mm")

    if "id" in lower_header:
        col_id = find_col(header, "id")
    else:
        col_id = None

    if "parent_id" in lower_header:
        col_parent = find_col(header, "parent_id")
    else:
        col_parent = None

    if "is_periodic_copy" in lower_header:
        col_copy = find_col(header, "is_periodic_copy")
    else:
        col_copy = None

    fibers = []

    for row in range(1, sheet.nrows):
        try:
            x = float(sheet.cell_value(row, col_x))
            y = float(sheet.cell_value(row, col_y))
            r = float(sheet.cell_value(row, col_r))
        except:
            continue

        if r <= 0.0:
            continue

        is_copy = False
        if col_copy is not None:
            is_copy = is_true_value(sheet.cell_value(row, col_copy))

        if is_copy and (not include_periodic_copies):
            continue

        if col_id is not None:
            try:
                fid = int(sheet.cell_value(row, col_id))
            except:
                fid = len(fibers) + 1
        else:
            fid = len(fibers) + 1

        if col_parent is not None:
            try:
                parent_id = int(sheet.cell_value(row, col_parent))
            except:
                parent_id = fid
        else:
            parent_id = fid

        # 保留所有可能与 RVE 盒子相交的周期纤维
        # 周期复制纤维的圆心可能略小于 0 或略大于 0.05，这是正常的。
        if (x + r < 0.0) or (x - r > Lx) or (y + r < 0.0) or (y - r > Ly):
            continue

        fibers.append({
            "id": fid,
            "parent_id": parent_id,
            "is_copy": is_copy,
            "x": x,
            "y": y,
            "r": r
        })

    return fibers


# ============================================================
# 3. Material and section
# ============================================================

def create_dummy_section(model):
    mat_name = "Fiber_dummy"
    sec_name = "Fiber_section_dummy"

    if mat_name not in model.materials.keys():
        mat = model.Material(name=mat_name)
        # 临时各向同性材料，后面可以替换成横观各向同性碳纤维
        mat.Elastic(table=((230000.0, 0.2),))

    if sec_name not in model.sections.keys():
        model.HomogeneousSolidSection(
            name=sec_name,
            material=mat_name,
            thickness=None
        )

    return sec_name


def assign_section_to_part(part, section_name):
    if section_name is None:
        return

    if len(part.cells) == 0:
        return

    region = regionToolset.Region(cells=part.cells[:])

    part.SectionAssignment(
        region=region,
        sectionName=section_name,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField="",
        thicknessAssignment=FROM_SECTION
    )


# ============================================================
# 4. Create fiber cylinders one by one
# ============================================================

def create_single_fiber_part(model, fiber, index, section_name=None):
    """
    每根纤维单独创建一个圆柱。
    局部圆心先放在 (0,0)，之后在 Assembly 中平移到 Excel 坐标。
    """
    r = fiber["r"]

    part_name = "Fiber_%04d" % index
    sketch_name = "Sketch_Fiber_%04d" % index

    sheet_size = max(0.02, 10.0 * r)

    sketch = model.ConstrainedSketch(
        name=sketch_name,
        sheetSize=sheet_size
    )

    sketch.CircleByCenterPerimeter(
        center=(0.0, 0.0),
        point1=(r, 0.0)
    )

    part = model.Part(
        name=part_name,
        dimensionality=THREE_D,
        type=DEFORMABLE_BODY
    )

    try:
        part.BaseSolidExtrude(
            sketch=sketch,
            depth=Lz
        )
    except Exception as e:
        print "------------------------------------------------------------"
        print "Failed to extrude fiber:"
        print "index     =", index
        print "fiber id  =", fiber["id"]
        print "parent id =", fiber["parent_id"]
        print "is copy   =", fiber["is_copy"]
        print "x = %.10f, y = %.10f, r = %.10f" % (fiber["x"], fiber["y"], fiber["r"])
        print "------------------------------------------------------------"
        raise e

    try:
        del model.sketches[sketch_name]
    except:
        pass

    assign_section_to_part(part, section_name)

    return part


def create_all_fiber_instances(model, fibers, section_name=None):
    """
    循环所有纤维，逐根创建圆柱并平移到对应坐标。
    """
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)

    instances = []

    for i, f in enumerate(fibers, start=1):

        part = create_single_fiber_part(
            model=model,
            fiber=f,
            index=i,
            section_name=section_name
        )

        inst_name = "Fiber_%04d-1" % i

        a.Instance(
            name=inst_name,
            part=part,
            dependent=ON
        )

        a.translate(
            instanceList=(inst_name,),
            vector=(f["x"], f["y"], 0.0)
        )

        instances.append(a.instances[inst_name])

        if i % 10 == 0:
            print "Created %d fiber cylinders..." % i

    return instances


def merge_fiber_instances(model, instances, section_name=None):
    """
    可选：把所有纤维实例合并成一个 Part。
    80% 体积分数下 Boolean merge 有可能比较慢或失败。
    """
    a = model.rootAssembly

    print "Merging fiber instances..."

    a.InstanceFromBooleanMerge(
        name="Fibers_Merged",
        instances=tuple(instances),
        keepIntersections=ON,
        originalInstances=SUPPRESS,
        domain=GEOMETRY
    )

    merged_part = model.parts["Fibers_Merged"]

    if len(merged_part.cells) > 0:
        merged_part.Set(cells=merged_part.cells[:], name="FIBERS_ALL")

    assign_section_to_part(merged_part, section_name)

    print "Merge finished."
    print "Merged fiber cells:", len(merged_part.cells)

    return merged_part


# ============================================================
# 5. Optional: create an RVE box for reference
# ============================================================

def create_rve_box_reference(model):
    """
    只创建一个透明参考盒子，用来检查纤维位置。
    如果你想完全手动建立基体，可以不用这个。
    """
    part_name = "RVE_Box_Reference"

    if part_name in model.parts.keys():
        del model.parts[part_name]

    s = model.ConstrainedSketch(
        name="Sketch_RVE_Box",
        sheetSize=max(Lx, Ly) * 2.0
    )

    s.rectangle(
        point1=(0.0, 0.0),
        point2=(Lx, Ly)
    )

    p = model.Part(
        name=part_name,
        dimensionality=THREE_D,
        type=DEFORMABLE_BODY
    )

    p.BaseSolidExtrude(
        sketch=s,
        depth=Lz
    )

    try:
        del model.sketches["Sketch_RVE_Box"]
    except:
        pass

    a = model.rootAssembly
    a.Instance(
        name=part_name + "-1",
        part=p,
        dependent=ON
    )

    return p


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":

    print "============================================================"
    print "Reading fiber data from Excel..."
    print "Excel file :", EXCEL_FILE
    print "Sheet name :", SHEET_NAME
    print "Include periodic copies:", INCLUDE_PERIODIC_COPIES
    print "============================================================"

    fibers = read_fiber_data_from_excel(
        EXCEL_FILE,
        SHEET_NAME,
        INCLUDE_PERIODIC_COPIES
    )

    print "Number of fiber cylinders to create:", len(fibers)

    if len(fibers) == 0:
        raise RuntimeError(
            "No fiber data was read. Please check Excel path, sheet name, and column names."
        )

    model = get_model(MODEL_NAME)

    delete_old_fibers(model)

    section_name = None
    if CREATE_DUMMY_SECTION:
        section_name = create_dummy_section(model)

    print "Creating fiber cylinders one by one..."
    instances = create_all_fiber_instances(
        model=model,
        fibers=fibers,
        section_name=section_name
    )

    print "All fiber cylinders have been created."
    print "Instance number:", len(instances)

    if MERGE_FIBERS:
        try:
            merge_fiber_instances(
                model=model,
                instances=instances,
                section_name=section_name
            )
        except Exception as e:
            print "WARNING: Boolean merge failed."
            print "Individual fiber instances are kept."
            print e
    else:
        print "MERGE_FIBERS = False."
        print "Fibers are kept as individual instances."

    # 可选：创建 RVE 参考盒子
    # 如果不需要，把下一行注释掉
    create_rve_box_reference(model)

    if SAVE_CAE:
        print "Saving CAE file..."
        mdb.saveAs(pathName=CAE_FILE)

    print "============================================================"
    print "Done."
    print "RVE box should be:"
    print "X: 0 to %.6f mm" % Lx
    print "Y: 0 to %.6f mm" % Ly
    print "Z: 0 to %.6f mm" % Lz
    print "============================================================"