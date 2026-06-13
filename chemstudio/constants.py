from __future__ import annotations


class MaterialTypeEnum:
    BASE_OIL = "base_oil"
    ADDITIVE = "additive"


class BaseOilCategory:
    MINERAL_OIL_PARAFFINIC = "mineral_oil_paraffinic"
    MINERAL_OIL_NAPHTHENIC = "mineral_oil_naphthenic"
    SYNTHETIC_PAO = "synthetic_pao"
    SYNTHETIC_ESTER = "synthetic_ester"
    SYNTHETIC_PAG = "synthetic_pag"
    SYNTHETIC_SILICONE = "synthetic_silicone"
    VEGETABLE_OIL = "vegetable_oil"


class AdditiveCategory:
    ANTIOXIDANT = "antioxidant"
    ANTIWEAR = "antiwear"
    EXTREME_PRESSURE = "extreme_pressure"
    FRICTION_MODIFIER = "friction_modifier"
    VISCOSITY_INDEX_IMPROVER = "viscosity_index_improver"
    CORROSION_INHIBITOR = "corrosion_inhibitor"
    DETERGENT = "detergent"
    DISPERSANT = "dispersant"
    POUR_POINT_DEPRESSANT = "pour_point_depressant"
    DEFOAMER = "defoamer"


class LubricantPropertyName:
    OXIDATION_ONSET_TEMP = "oxidation_onset_temperature"
    EXTREME_PRESSURE_LOAD = "extreme_pressure_load"
    WELD_LOAD = "weld_load"
    AVERAGE_FRICTION_COEFF = "average_friction_coefficient"
    WEAR_SCAR_WIDTH = "wear_scar_width"
    WEAR_SCAR_DEPTH = "wear_scar_depth"
    WEAR_SPOT_DIAMETER = "wear_spot_diameter"
    KINEMATIC_VISCOSITY_40C = "kinematic_viscosity_40c"
    KINEMATIC_VISCOSITY_100C = "kinematic_viscosity_100c"
    VISCOSITY_INDEX = "viscosity_index"
    POUR_POINT = "pour_point"
    FLASH_POINT = "flash_point"
    TOTAL_ACID_NUMBER = "total_acid_number"


class FormulationRole:
    BASE_OIL = "base_oil"
    ADDITIVE = "additive"


LUBRICANT_PROPERTY_UNITS = {
    LubricantPropertyName.OXIDATION_ONSET_TEMP: "degC",
    LubricantPropertyName.EXTREME_PRESSURE_LOAD: "N",
    LubricantPropertyName.WELD_LOAD: "N",
    LubricantPropertyName.AVERAGE_FRICTION_COEFF: "",
    LubricantPropertyName.WEAR_SCAR_WIDTH: "mm",
    LubricantPropertyName.WEAR_SCAR_DEPTH: "um",
    LubricantPropertyName.WEAR_SPOT_DIAMETER: "mm",
    LubricantPropertyName.KINEMATIC_VISCOSITY_40C: "cSt",
    LubricantPropertyName.KINEMATIC_VISCOSITY_100C: "cSt",
    LubricantPropertyName.VISCOSITY_INDEX: "",
    LubricantPropertyName.POUR_POINT: "degC",
    LubricantPropertyName.FLASH_POINT: "degC",
    LubricantPropertyName.TOTAL_ACID_NUMBER: "mg KOH/g",
}


MATERIAL_TYPE_ALIASES = {
    "基础油": MaterialTypeEnum.BASE_OIL,
    "base oil": MaterialTypeEnum.BASE_OIL,
    "base_oil": MaterialTypeEnum.BASE_OIL,
    "添加剂": MaterialTypeEnum.ADDITIVE,
    "additive": MaterialTypeEnum.ADDITIVE,
    "石蜡基矿物油": "mineral_oil",
    "环烷基矿物油": "mineral_oil",
    "矿物油": "mineral_oil",
    "mineral_oil": "mineral_oil",
    "合成油": "synthetic",
    "synthetic": "synthetic",
    "聚α-烯烃": "synthetic",
    "pao": "synthetic",
    "合成酯": "synthetic",
    "ester": "synthetic",
    "聚醚": "synthetic",
    "pag": "synthetic",
    "硅油": "synthetic",
    "silicone": "synthetic",
    "植物油": "vegetable",
    "天然酯": "vegetable",
    "vegetable": "vegetable",
    "抗氧剂": AdditiveCategory.ANTIOXIDANT,
    "antioxidant": AdditiveCategory.ANTIOXIDANT,
    "抗磨剂": AdditiveCategory.ANTIWEAR,
    "antiwear": AdditiveCategory.ANTIWEAR,
    "极压剂": AdditiveCategory.EXTREME_PRESSURE,
    "extreme_pressure": AdditiveCategory.EXTREME_PRESSURE,
    "摩擦改进剂": AdditiveCategory.FRICTION_MODIFIER,
    "friction_modifier": AdditiveCategory.FRICTION_MODIFIER,
    "粘度指数改进剂": AdditiveCategory.VISCOSITY_INDEX_IMPROVER,
    "viscosity_index_improver": AdditiveCategory.VISCOSITY_INDEX_IMPROVER,
    "缓蚀剂": AdditiveCategory.CORROSION_INHIBITOR,
    "corrosion_inhibitor": AdditiveCategory.CORROSION_INHIBITOR,
    "清净剂": AdditiveCategory.DETERGENT,
    "detergent": AdditiveCategory.DETERGENT,
    "分散剂": AdditiveCategory.DISPERSANT,
    "dispersant": AdditiveCategory.DISPERSANT,
    "降凝剂": AdditiveCategory.POUR_POINT_DEPRESSANT,
    "pour_point_depressant": AdditiveCategory.POUR_POINT_DEPRESSANT,
    "抗泡剂": AdditiveCategory.DEFOAMER,
    "defoamer": AdditiveCategory.DEFOAMER,
}
