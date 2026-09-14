import re

def extract_fast_features(request_dict):
    """
    Table I: 9 Fast Features.
    Ported from synthetic prototype.
    """
    path = request_dict.get("path", "")
    method = request_dict.get("method", "GET")
    headers = request_dict.get("headers", {})
    body = request_dict.get("body", "")
    
    # 1. request_length: total length of path + headers + body
    req_len = len(path) + sum(len(k) + len(str(v)) for k, v in headers.items()) + len(body)
    
    # 2. num_headers
    num_headers = len(headers)
    
    # 3. special_chars_count (in path and body)
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', path + body))
    
    # 4. sql_keywords_count
    sql_kws = ["SELECT", "UNION", "DROP", "INSERT", "DELETE", "UPDATE", "OR 1=1"]
    sql_count = sum(1 for kw in sql_kws if kw.lower() in (path + body).lower())
    
    # 5. script_tags_count (XSS)
    script_count = len(re.findall(r'<script', (path + body).lower()))
    
    # 6. path_depth
    path_depth = path.count('/')
    
    # 7. is_post
    is_post = 1 if method.upper() == "POST" else 0
    
    # 8. user_agent_len
    ua_len = len(str(headers.get("User-Agent", "")))
    
    # 9. consecutive_symbols (e.g., ../ or %20%20)
    consec_sym = len(re.findall(r'[^a-zA-Z0-9\s]{2,}', path + body))
    
    return [
        req_len,
        num_headers,
        special_chars,
        sql_count,
        script_count,
        path_depth,
        is_post,
        ua_len,
        consec_sym
    ]
