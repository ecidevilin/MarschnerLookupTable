from skimage import io, data
from numpy import *
import sys

resolution = 128
refraction = 1.55   # 1.0 , 3.0
absorptionAsColor = False
absorption = array([0.0, 0.0, 0.0])
eccentricity = 1.0 # 0.5 , 1.0
shiftR = -8.0 # -20.0 , 5.0
shiftTT = 4.0 # -5.0 , 20.0
shiftTRT = 12.0 # -5.0 , 20.0
widthR = 10.0 # 0.0 , 45.0
widthTT = 5.0 # 0.0, 45.0
widthTRT = 20.0 # 0.0, 45.0
glint = 1.0 # 0.0 , 10.0
causticWidth = 20.0 # 0.0 , 45.0
causticFade = 0.2 # 0.01 , 0.5
causticLimit = 0.5 # 0.0, 10.0

shiftR *= pi / 180.0
shiftTT *= pi / 180.0
shiftTRT *= pi / 180.0
widthR *= pi / 180.0
widthTT *= pi / 180.0
widthTRT *= pi / 180.0
causticWidth *= pi / 180.0

#####################################################################################################################

def marschnerM(shift, width, normWidth, x):
    norm = 1.0 / ( ((normWidth/180.0)* pi) * sqrt( 2.0 * pi ) )
    coefficients = gaussianPDF(shift, width)
    return ( gaussian( coefficients[0], coefficients[1], coefficients[2], x ) / coefficients[0] ) * norm

def gaussian(a, b, c, x):
    o = x - b
    return a * exp(- (o * o) / (2.0 * c * c))

def gaussianPDF(mu, sigma):
    return array([1.0/(sigma * sqrt( 2.0 * pi )), mu, sigma])

def bravaisIndex(theta, eta):
    sinTheta = sin(theta)
    return sqrt(eta * eta - sinTheta * sinTheta) / cos(theta)

def fresnel(incidenceAngle, etaPerp, etaParal, invert):
    n1 = 0.0
    n2 = 0.0

    rPerp = 1.0
    rParal = 1.0

    angle = fabs(incidenceAngle)
    if (angle > pi / 2.0):
        angle = pi - angle
    cosA = cos(angle)
    if (invert):
        n1 = etaPerp
        n2 = 1.0
    else:
        n1 = 1.0
        n2 = etaPerp

    a = (n1 / n2) * sin(angle)
    a *= a
    if (a <= 1.0):
        b = n2 * sqrt(1.0 - a)
        c = n1 * cosA
        rPerp = (c - b) / (c + b)
        rPerp *= rPerp
        rPerp = min(1.0, rPerp)
    if (invert):
        n1 = etaParal
        n2 = 1.0
    else:
        n1 = 1.0
        n2 = etaParal

    d = n1 / n2 * sin(angle)
    d *= d
    if (d <= 1.0):
        e = n1 * sqrt(1 - d)
        f = n2 * cosA
        rParal = (e - f) / (e + f)
        rParal *= rParal
        rParal = min(1.0, rParal)
    return 0.5 * (rPerp + rParal)

def eccentricityRefraction(averageAzimuth ):
    n1 = 2.0 * (refraction - 1.0) * eccentricity * eccentricity - refraction + 2.0
    n2 = 2.0 * (refraction - 1.0) / (eccentricity * eccentricity) - refraction + 2.0
    return ((n1 + n2) + cos(2.0 * averageAzimuth) * (n1 - n2)) / 2.0


def exitAngle(p, eta, h):
    # use polynomial that approximates original equation.
    gamma = arcsin(h)
    c = arcsin(1.0 / eta);
    return (6.0 * p * c / pi - 2.0) * gamma - 8.0 * (p * c / (pi * pi * pi)) * gamma * gamma * gamma + p * pi;

def dExitAngle(p, eta, h):
    gamma = arcsin(h)
    c = arcsin(1.0/eta)
    dGamma = (6.0*p*c/pi-2.0)-3.0*8.0*(p*c/(pi*pi*pi) )*gamma*gamma
    denom = sqrt(1.0 - h * h)
    return dGamma / max(1e-5, denom)

def ddExitAngle(p, eta, h):
    # computes the second derivative of the polynomial relative to h.
    gamma = arcsin(h)
    c = arcsin(1.0 / eta)
    dGamma = -2.0 * 3.0 * 8.0 * (p * c / (pi * pi * pi)) * gamma
    denom = pow(1.0 - h * h, 3.0 / 2.0)
    return (dGamma * h) / max(1e-5, denom)

def marschnerA(p, gammaI, refraction, etaPerp, etaParal, light):
    if (p == 0):
        return fresnel( gammaI, etaPerp, etaParal, False)
    h = sin(gammaI)
    gammaT = arcsin( clamp( h/etaPerp, -1.0, 1.0 ) )
    etaOverRcosY = clamp((etaPerp / refraction) * cos(light[1]), -1.0, 1.0)
    thetaT = arccos(etaOverRcosY)
    cosTheta = cos(thetaT)
    l = 2.0 * cos(gammaT) / cosTheta
    segmentAbsorption = array([0.0,0.0,0.0])
    for i in range(0, 3):
        segmentAbsorption[i] = exp(-absorption[i] * l * p)
    invFr = fresnel(gammaT, etaPerp, etaParal, True)
    fr = (1.0 - fresnel(gammaI, etaPerp, etaParal, False)) * (1.0 - invFr)
    if (p > 1.0):
        fr *= invFr
    return fr * segmentAbsorption

def targetAngle(p, relativeAzimuth ):
    t = fabs(relativeAzimuth)
    if (p != 1):
        if (t > pi):
            t -= 2.0 * pi
        t += p * pi
    return t

def marschnerNP(p, refraction, etaPerp, etaParal, light, targetAngle):
    C = arcsin(clamp(1.0 / etaPerp, -1.0, 1.0))
    a = -8.0 * (p * C / (pi * pi * pi))
    b = 0.0
    c = ((6.0 * p * C) / pi) - 2.0
    d = (p * pi) - targetAngle

    roots = array([0.0,0.0,0.0])
    rootCount = cubicRoots(a, b, c, d, roots)

    result = array([0.0,0.0,0.0])
    for i in range(0, rootCount):
        gammaI = roots[i]
        if (fabs(gammaI) <= pi / 2.0):
            h = sin(gammaI)
            finalAbsorption = marschnerA( p, gammaI, refraction, etaPerp, etaParal, light )
            denom = max(1e-5, ( 2.0 * fabs( dExitAngle( p, etaPerp, h ) ) ))
            result += finalAbsorption / denom
    return result


def marschnerNTRT(refraction, etaPerp, etaParal, light, targetAngle):
    dH = 0.0
    t = 0.0
    hc = 0.0
    Oc1 = 0.0
    Oc2 = 0.0
    if (etaPerp < 2.0):
        c = arcsin(1.0 / etaPerp)
        gammac = sqrt(((6.0 * 2.0 * c) / (pi - 2.0)) / (3.0 * 8.0 * ((2.0 * c) / (pi * pi * pi))))
        hc = fabs(sin(gammac))
        ddexitAngle = ddExitAngle(2, etaPerp, hc)
        dH = min(causticLimit, 2.0 * sqrt((2.0 * causticWidth) / fabs(ddexitAngle)))
        t = 1.0
    else:
        hc = 0.0
        dH = causticLimit
        t = 1.0 - smoothstep(2.0, 2.0 + causticFade, etaPerp)
    Oc1 = exitAngle(2, etaPerp, hc)
    Oc2 = exitAngle(2, etaPerp, -hc)
    coefficients = gaussianPDF(0.0, causticWidth)
    causticCenter = gaussian(coefficients[0], coefficients[1], coefficients[2], 0.0)
    causticLeft = gaussian(coefficients[0], coefficients[1], coefficients[2], targetAngle - Oc1)
    causticRight = gaussian(coefficients[0], coefficients[1], coefficients[2], targetAngle - Oc2)
    A = marschnerA(2, arcsin(hc), refraction, etaPerp, etaParal, light)
    L = marschnerNP(2, refraction, etaPerp, etaParal, light, targetAngle)
    L *= 1.0 - t * causticLeft / causticCenter
    L *= 1.0 - t * causticRight / causticCenter

    L += t * glint * A * dH * (causticLeft + causticRight)
    return L

def MarschnerMR(eye, light):
    rWidth = 5.0
    averageTheta = (eye[1] + light[1]) / 2.0
    relativeTheta = fabs(eye[1] - light[1]) / 2.0
    cosRelativeTheta =cos(relativeTheta)
    invSqrCosRelativeTheta = 1.0 / (cosRelativeTheta * cosRelativeTheta)
    cosLight = cos(light[1])
    finalScale = invSqrCosRelativeTheta * cosLight

    return marschnerM( shiftR, widthR, rWidth, averageTheta ) * finalScale

def MarschnerMTT(eye, light):
    rWidth = 5.0
    averageTheta = (eye[1] + light[1]) / 2.0
    relativeTheta = fabs(eye[1] - light[1]) / 2.0
    cosRelativeTheta = cos(relativeTheta)
    invSqrCosRelativeTheta = 1.0 / (cosRelativeTheta * cosRelativeTheta)
    cosLight = cos(light[1])
    finalScale = invSqrCosRelativeTheta * cosLight

    return marschnerM(shiftTT, widthTT, rWidth / 2.0, averageTheta) * finalScale

def MarschnerMTRT(eye, light):
    rWidth = 5.0
    averageTheta = (eye[1] + light[1]) / 2.0
    relativeTheta = fabs(eye[1] - light[1]) / 2.0
    cosRelativeTheta = cos(relativeTheta)
    invSqrCosRelativeTheta = 1.0 / (cosRelativeTheta * cosRelativeTheta)
    cosLight = cos(light[1])
    finalScale = invSqrCosRelativeTheta * cosLight

    return marschnerM(shiftTRT, widthTRT, rWidth * 2.0, averageTheta) * finalScale


def MarschnerNR(eye, light):
    relativeTheta = fabs(eye[1] - light[1]) / 2.0
    relativeAzimuth = fmod(fabs(eye[0] - light[0]), 2.0 * pi)
    etaPerp = bravaisIndex(relativeTheta, refraction)
    etaParal = (refraction * refraction) / etaPerp

    return marschnerNP(0, refraction, etaPerp, etaParal, light, targetAngle(0, relativeAzimuth))

def MarschnerNTT(eye, light):
    relativeTheta = fabs(eye[1] - light[1]) / 2.0
    relativeAzimuth = fmod(fabs(eye[0] - light[0]), 2.0 * pi)
    etaPerp = bravaisIndex(relativeTheta, refraction)
    etaParal = (refraction * refraction) / etaPerp

    return marschnerNP(1, refraction, etaPerp, etaParal, light, targetAngle(1, relativeAzimuth))

def MarschnerNTRT(eye, light):
    relativeTheta = fabs(eye[1] - light[1]) / 2.0
    relativeAzimuth = fmod(fabs(eye[0] - light[0]), 2.0 * pi)
    refractionTRT = eccentricityRefraction((eye[0] + light[0]) / 2.0)
    etaPerpTRT = bravaisIndex(relativeTheta, refractionTRT)
    etaParalTRT = (refractionTRT * refractionTRT) / etaPerpTRT

    return marschnerNTRT(refractionTRT, etaPerpTRT, etaParalTRT, light, targetAngle(2, relativeAzimuth))

def cubicRoots(A, B, C, D, roots):
    if (fabs(A) < sys.float_info.epsilon):
        return quadraticRoots( B, C, D, roots)
    else:
        return normalizedCubicRoots(B / A, C / A, D / A, roots)


def normalizedCubicRoots(A, B, C, roots):
    if (fabs(C) < sys.float_info.epsilon):
        return quadraticRoots( 1, A, B, roots)
    Q = (3.0 * B - A * A) / 9.0
    R = (9.0 * A * B - 27.0 * C - 2.0 * A * A * A) / 54.0
    D = Q * Q * Q + R * R

    if (D >= 0):
        sqrtD = sqrt(D)
        S = cubicRoot(R + sqrtD)
        t = cubicRoot(R - sqrtD)
        roots[0] = -A / 3.0 + (S + t)
        return 1
    else:
        th = arccos(R / sqrt(-Q * Q * Q))
        sqrtQ = sqrt(-Q)
        roots[0] = 2.0 * sqrtQ * cos(th / 3.0) - A / 3.0
        roots[1] = 2.0 * sqrtQ * cos((th + 2.0 * pi) / 3.0) - A / 3.0
        roots[2] = 2.0 * sqrtQ * cos((th + 4.0 * pi) / 3.0) - A / 3.0
        return 3

def cubicRoot(v):
    if (v < 0):
        return -pow(-v, 1.0 / 3.0)
    return pow(v, 1.0 / 3.0)

def quadraticRoots(a, b, c, roots):
    if (fabs(a) < sys.float_info.epsilon):
        return linearRoots(b, c, roots)
    D = b * b - 4.0 * a * c
    if (fabs(D) < sys.float_info.epsilon):
        roots[0] = -b / (2.0 * a)
        return 1
    elif D > 0:
        s = sqrt(D)
        roots[0] = (-b + s) / (2.0 * a)
        roots[1] = (-b - s) / (2.0 * a)
        return 2
    return 0

def linearRoots(a, b, roots):
    rootCount = -1;
    if (a != 0.0):
        roots[0] = -b / a
        rootCount = 1
    elif (b != 0.0):
        rootCount = 0
    return rootCount

def clamp(v, mimv, maxv):
    return max(mimv, min(v, maxv))


def smoothstep( v0, v1, v ):
    x = (v-v0)/(v1-v0)
    if ( x > 0 ):
        if ( x < 1 ):
            return (3-2*x)*x*x
        return 1
    return 0





cosDiffTheta = ndarray((resolution, resolution, 1), float)
mr = ndarray((resolution, resolution, 1), float)
mtt = ndarray((resolution, resolution, 1), float)
mtrt = ndarray((resolution, resolution, 1), float)
nr = ndarray((resolution, resolution, 1), float)
ntt = ndarray((resolution, resolution, 3), float)
ntrt = ndarray((resolution, resolution, 3), float)

light = array([0.0, 0.0, 0.0])
eye = array([0.0, 0.0, 0.0])

step = 2.0 / resolution
sinqo = -1.0
sinqi = -1.0
qd = 0.0

for y in range(0, resolution):
    eye[1] = arcsin(sinqo)
    sinqi = -1.0
    for x in range(0, resolution):
        light[1] = arcsin(sinqi)
        qd = (light[1] - eye[1]) / 2.0
        cosDiffTheta[x,y] = (1.0 + cos(qd)) / 2.0
        mr[x,y] = clamp(MarschnerMR(eye, light) / 30.0, 0, 1)
        mtt[x,y] = clamp(MarschnerMTT(eye, light) / 30.0, 0, 1)
        mtrt[x,y] = clamp(MarschnerMTRT(eye, light) / 30.0, 0, 1)
        sinqi += step
    sinqo += step

cosfd = -1.0
cosqd = -1.0
light[0] = 0.0
relativeTheta = 0.0
for y in range(0, resolution):
    relativeTheta = arccos(cosqd)
    light[1] = relativeTheta / 2.0
    eye[1] = -relativeTheta / 2.0
    cosfd = -1.0
    for x in range(0, resolution):
        eye[0] = arccos(cosfd)
        nr[x,y] = MarschnerNR(eye, light)[0]
        ntt[x,y] = MarschnerNTT(eye, light)
        ntrt[x,y] = MarschnerNTRT(eye, light)
        cosfd += step
    cosqd += step

lut0 = ndarray((resolution, resolution, 4), float)
lut0[:, :, :0] = mr
lut0[:, :, 1:2] = mtt
lut0[:, :, 2:3] = mtrt
lut0[:, :, 3:] = cosDiffTheta
lut1 = ndarray((resolution, resolution, 4), float)
lut1[:, :, :3] = ntrt
lut1[:, :, 3:] = nr

io.imsave('d://lut0.png', lut0)
io.imsave('d://lut1.png', lut1)
