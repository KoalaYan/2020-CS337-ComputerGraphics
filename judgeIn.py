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
    if slat==elat: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if slat>lat and elat>lat: #线段在射线上边
        return False
    if slat<lat and elat<lat: #线段在射线下边
        return False
    if slat==lat and elat>lat: #交点为下端点，对应spoint
        return False
    if elat==lat and slat>lat: #交点为下端点，对应epoint
        return False
    if slng<lng and elat<lat: #线段在射线左边
        return False
    #求交点
    xseg=elng-(elng-slng)*(elat-lat)/(elat-slat)
    if xseg == lng:
        #print("点在多边形的边上")
        return None
    if xseg<lng: #交点在射线起点的左侧
        return False
    return True  #排除上述情况之后

def isin_multipolygon(poi,vertex_lst, contain_boundary=True):
    # 判断是否在外包矩形内，如果不在，直接返回false
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




if __name__ == '__main__':
    vertex_lst = [[0,0],[1,1],[1,2],[0,2],[0,0]]
    poi = [0.5,1.75]
    print(isin_multipolygon(poi,vertex_lst, contain_boundary=True))