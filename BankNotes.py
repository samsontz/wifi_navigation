from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    NMAIST_D: int
    NMAIST_B: int
    NMAIST_C: int
    CDAC_14: int