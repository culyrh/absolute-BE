# app/utils/address_utils.py

def extract_sidocode(adm_cd: str | int) -> str:
    """
    adm_cd (행정동 코드 8자리)에서 시도코드 2자리 추출
    """
    adm_cd = str(adm_cd)
    if len(adm_cd) < 2:
        return None
    return adm_cd[:2]
