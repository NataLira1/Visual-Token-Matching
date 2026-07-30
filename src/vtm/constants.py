"""IDs e splits publicados pelo Taskonomy."""

TASKONOMY_CLASSES = {
    "background": 1,
    "bottle": 2,
    "chair": 3,
    "couch": 4,
    "plant": 5,
    "bed": 6,
    "dining_table": 7,
    "toilet": 8,
    "tv": 9,
    "microwave": 10,
    "oven": 11,
    "toaster": 12,
    "sink": 13,
    "refrigerator": 14,
    "book": 15,
    "clock": 16,
    "vase": 17,
}

TINY_BUILDINGS = {
    "train": (
        "hanson", "merom", "klickitat", "onaga", "leonardo", "marstons",
        "newfields", "pinesdale", "lakeville", "cosmos", "benevolence",
        "pomaria", "tolstoy", "shelbyville", "allensville", "wainscott",
        "beechwood", "coffeen", "stockman", "hiteman", "woodbine",
        "lindenwood", "forkland", "mifflinburg", "ranchester",
    ),
    "val": ("wiconisco", "corozal", "collierville", "markleeville", "darden"),
    "test": ("ihlen", "muleshoe", "uvalda", "noxapater", "mcdade"),
}

BUILDING_TO_SPLIT = {
    building: split
    for split, buildings in TINY_BUILDINGS.items()
    for building in buildings
}
