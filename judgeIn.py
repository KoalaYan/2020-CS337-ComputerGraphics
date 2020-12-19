def isinpolygon(point,vertex_lst:list, contain_boundary=True):
    lngaxis, lataxis = zip(*vertex_lst)
    minlng, maxlng = min(lngaxis),max(lngaxis)
    minlat, maxlat = min(lataxis),max(lataxis)
    lng, lat = point
    if contain_boundary:
        isin = (minlng<=lng<=maxlng) & (minlat<=lat<=maxlat)
    else:
        isin = (minlng<lng<maxlng) & (minlat<lat<maxlat)
    return isin

def isintersect(poi,spoi,epoi):
    lng, lat = poi
    slng, slat = spoi
    elng, elat = epoi
    if poi == spoi:
        return None
    if slat==elat:
        return False
    if slat>lat and elat>lat: # above the ray
        return False
    if slat<lat and elat<lat: # below the ray
        return False
    if slat==lat and elat>lat: # lower endpoint, corresponding to spoint
        return False
    if elat==lat and slat>lat: # lower endpoint, corresponding to epoint
        return False
    if slng<lng and elat<lat: # left of the ray
        return False
    # intersection
    xseg=elng-(elng-slng)*(elat-lat)/(elat-slat)
    if xseg == lng:
        # Point on the edge of the polygon
        return None
    if xseg<lng:
        # The intersection is to the left of the beginning of the ray
        return False
    return True

def isin_multipolygon(poi,vertex_lst, contain_boundary=True):
    # in the outer rectangle ? if not, return false directly
    if not isinpolygon(poi, vertex_lst, contain_boundary):
        return False
    sinsc = 0
    for spoi, epoi in zip(vertex_lst[:-1],vertex_lst[1::]):
        intersect = isintersect(poi, spoi, epoi)
        if intersect is None:
            return (False, True)[contain_boundary]
        elif intersect:
            sinsc+=1
    return sinsc%2==1
