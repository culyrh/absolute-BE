# app/utils/address_utils.py

def extract_sidocode(adm_cd2: str | int) -> str:
    """
    adm_cd2 (법정동 코드 10자리)에서 시도코드 2자리 추출
    """
    adm_cd2 = str(adm_cd2)
    if len(adm_cd2) < 2:
        return None
    return adm_cd2[:2]
