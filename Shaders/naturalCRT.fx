
// Oct.15.2021 (hotfix)

#ifndef MAX_SOURCE_SIZE_X
    #define MAX_SOURCE_SIZE_X 1024
#endif
#ifndef MAX_SOURCE_SIZE_Y
    #define MAX_SOURCE_SIZE_Y 1024
#endif

/**********************************************************
 *  inputs
 **********************************************************/

#define SLID(n, m, M) ui_type="slider"; ui_label=n; ui_min=m; ui_max=M
#define DRAG(n, m)    ui_type="drag"; ui_label=n; ui_min=m
#define COMB(n,i)     ui_type="combo"; ui_label=n; ui_items=i
#define RADI(n,i)     ui_type="radio"; ui_label=n; ui_items=i
#define NAME(n)       ui_label=n

// texture lvs
#define SCANLINE_LV0_INDEX 0
#define SCANLINE_LV0 "scanine_lv0\0"
#define SCANLINE_LV1_INDEX 1
#define SCANLINE_LV1 "scanine_lv1\0"
#define SCANLINE_LV2_INDEX 2,3,4
#define SCANLINE_LV2 "scanline_lv2\0scanline_lv2_thin\0scanline_lv2_beamoffset\0"

#define PHOSPHOR_LV0_INDEX 0
#define PHOSPHOR_LV0 "slotmask_lv0\0"
#define PHOSPHOR_LV1_INDEX 1,4
#define PHOSPHOR_LV1 "slotmask_lv1\0aperturegrille_lv1\0"
#define PHOSPHOR_LV2_INDEX 2,5
#define PHOSPHOR_LV2 "slotmask_lv2\0aperturegrille_lv2\0"
#define PHOSPHOR_LV3_INDEX 3,6,7
#define PHOSPHOR_LV3 "slotmask_lv3\0aperturegrille_lv3\0shadowmask_lv3\0"

uniform bool   gPortrait   <NAME("Portrait mode");                              > = false;
uniform uint   gScanlineLV <COMB("Scanline LV", "Lv0\0Lv1\0Lv2\0Auto\0");       > = 3;
uniform uint   gPhosphorLV <COMB("Phosphor LV", "Lv0\0Lv1\0Lv2\0Lv3\0Auto\0");  > = 4;

#define SCANLINE ui_category="Scanline"; ui_category_closed=true
uniform uint   gScanline0 <SCANLINE; COMB("Scanline Lv0", SCANLINE_LV0); > = 0;
uniform uint   gScanline1 <SCANLINE; COMB("Scanline Lv1", SCANLINE_LV1); > = 0;
uniform uint   gScanline2 <SCANLINE; COMB("Scanline Lv2", SCANLINE_LV2); > = 0;
static const uint gScanline0ID[] = { SCANLINE_LV0_INDEX };
static const uint gScanline1ID[] = { SCANLINE_LV1_INDEX };
static const uint gScanline2ID[] = { SCANLINE_LV2_INDEX };

#define PHOSPHOR ui_category="Phosphor coating"; ui_category_closed=true
uniform uint   gPhosphor0 <PHOSPHOR; COMB("Phosphor Lv0", PHOSPHOR_LV0); > = 0;
uniform uint   gPhosphor1 <PHOSPHOR; COMB("Phosphor Lv1", PHOSPHOR_LV1); > = 0;
uniform uint   gPhosphor2 <PHOSPHOR; COMB("Phosphor Lv2", PHOSPHOR_LV2); > = 0;
uniform uint   gPhosphor3 <PHOSPHOR; COMB("Phosphor Lv3", PHOSPHOR_LV3); > = 0;
uniform uint2  gLoopSize0 <PHOSPHOR; DRAG("Loop Size Lv0", 1); > = (4).xx;
uniform uint2  gLoopSize1 <PHOSPHOR; DRAG("Loop Size Lv1", 1); > = (4).xx;
uniform uint2  gLoopSize2 <PHOSPHOR; DRAG("Loop Size Lv2", 1); > = (6).xx;
uniform uint2  gLoopSize3 <PHOSPHOR; DRAG("Loop Size Lv3", 1); > = (8).xx;
static const uint gPhosphor0ID[] = { PHOSPHOR_LV0_INDEX };
static const uint gPhosphor1ID[] = { PHOSPHOR_LV1_INDEX };
static const uint gPhosphor2ID[] = { PHOSPHOR_LV2_INDEX };
static const uint gPhosphor3ID[] = { PHOSPHOR_LV3_INDEX };

uniform float  gIntensity  <SLID("Intensity", 0, 2 );      > = .5;
uniform float  gSaturation <SLID("Saturation", -1,1);      > = .5;
uniform float  gDiverge    <SLID("Divergence", 0,5);       > = .5;

uniform uint   gAutoRes    <COMB("Auto Target Res", "None\0Over\0Under\0");> = 0;
uniform bool   gAspect     <NAME("Auto Target Keep Aspect"); > = true;

uniform uint   gSrcResX    <SLID("Source Size X", 16, MAX_SOURCE_SIZE_X); > = 256; // unused when auto
uniform uint   gSrcResY    <SLID("Source Size Y", 16, MAX_SOURCE_SIZE_Y);> = 256;
uniform uint   gDesResX    <SLID("Target Size X", 16, BUFFER_WIDTH); > = 256;
uniform uint   gDesResY    <SLID("Target Size Y", 16, BUFFER_HEIGHT);> = 256;
uniform uint   gOutResX    <SLID("Output Size X", 16, BUFFER_WIDTH); > = BUFFER_WIDTH;
uniform uint   gOutResY    <SLID("Output Size Y", 16, BUFFER_HEIGHT);> = BUFFER_HEIGHT;

uniform uint   gMapping    <COMB("View Mode", "None\0Fill\0Contain\0Cover\0");> = 0;

uniform bool   gFlipX      <NAME("Flip X");                > = false;
uniform bool   gFlipY      <NAME("Flip Y");                > = false;
uniform float2 gScanScale  <SLID("Scan scale", .1, 10);    > = 1;
uniform float  gHBlur      <SLID("Horizontal Blur", 0, 1); > = 0;
uniform int    gIteration  <SLID("Iteration", 1, 10);      > = 1;

uniform bool   gCrosstalk  <NAME("Use Lum/Col Crosstalk"); > = true; // global switch
uniform float  gLumCross   <SLID("Lum Crosstalk", -1,1);   > = .5;
uniform float  gColCross   <SLID("Col Crosstalk", -1,1);   > = .5;

uniform float3 gGain       <SLID("Color Gain", -1,1);      > = (1).xxx;
uniform float  gGamma      <SLID("Gamma", 1e-3f, 3);       > = 1;

uniform int    gFramecount <source="framecount"; >;

/**********************************************************
 *  Texture Setup
 **********************************************************/

#define FILTER(a)       MagFilter = a; MinFilter = a; MipFilter = a
#define ADDRESS(a)      AddressU = a; AddressV = a; AddressW = a
#define BUFFER_SIZE     int2(BUFFER_WIDTH, BUFFER_HEIGHT)
#define BUFFER_FSIZE    float2(BUFFER_WIDTH, BUFFER_HEIGHT)
#define BUFFER_SCANW    (BUFFER_WIDTH + MAX_SOURCE_SIZE_X)
#define BUFFER_SCANH    (BUFFER_HEIGHT + MAX_SOURCE_SIZE_Y)
#define BUFFER_SCAN     int2(BUFFER_SCANW, BUFFER_SCANH)
#define BUFFER_FSCAN    float2(BUFFER_SCANW, BUFFER_SCANH)

int2    ScreenResolution() { return BUFFER_SIZE; }
int2    SourceResolution() { return max(16, uint2(gSrcResX, gSrcResY)); }
int2    TargetResolution() {
    int2 source = SourceResolution();
    // target cannot be smaller then source
    if(gAutoRes == 0) return max(source + 1, uint2(gDesResX, gDesResY));
    // auto res
    float2 scale = (BUFFER_FSIZE / source) + 0.0001;

    scale = gAutoRes == 1? ceil(scale) : floor(scale);

    return source * (gAspect? min(scale.xx,scale.yy) : scale);
}
int2   OutputResolution() { return int2(gOutResX, gOutResY); } // output to screen
float2 NormOutputRes() { return OutputResolution() / BUFFER_FSIZE; } // output to screen
// norm sizes

texture2D tColor : COLOR;
texture2D tBuffA { Width=BUFFER_WIDTH; Height=BUFFER_HEIGHT/2; };
texture2D tBuffB { Width=BUFFER_WIDTH; Height=BUFFER_HEIGHT/2; };
texture2D tScan  { Width=BUFFER_SCANW; Height=BUFFER_SCANH; };

sampler2D sLinear{ Texture=tColor; ADDRESS(BORDER); };
sampler2D sPoint { Texture=tColor; FILTER(POINT); };
sampler2D sScan  { Texture=tScan;  ADDRESS(BORDER);};
sampler2D sBuffA { Texture=tBuffA; ADDRESS(BORDER);}; // bilinear
sampler2D sBuffB { Texture=tBuffB; ADDRESS(BORDER);}; // bilinear

/**********************************************************
 *  sprite selector
 *      Allows select sprites dynamically.
 **********************************************************/
#define SPRITE_WIDTH    26
#define SPRITE_HEIGHT   35

texture2D tSprite < source="naturalCRT_sprites.png"; > { Width=SPRITE_WIDTH; Height=SPRITE_HEIGHT; };
sampler2D sSpriteP { Texture=tSprite; FILTER(POINT); };
sampler2D sSprite  { Texture=tSprite; FILTER(LINEAR); };

//note: add scanline padding for items.
static const int4 ScanlineRects[] = {
    int4(17,31,19,33),  // scanline_lv0.png
    int4(21,31,25,34),  // scanline_lv1.png
    int4(21,19,25,23),  // scanline_lv2.png
    int4(21,25,25,29),  // scanline_lv2_thin.png
    int4( 1,21, 9,29),  // scanline_lv2_beamoffset.png
};
static const int4 PhosphorRects[] = {
    int4(21, 7,25,11),  // slotmask_lv0.png
    int4(21,13,25,17),  // slotmask_lv1.png
    int4( 1,11, 9,19),  // slotmask_lv2.png
    int4( 1, 1, 9, 9),  // slotmask_lv3.png
    int4(21, 1,25, 5),  // aperturegrille_lv1.png
    int4(11, 1,19, 9),  // aperturegrille_lv2.png
    int4(11,11,19,19),  // aperturegrille_lv3.png
    int4(11,21,19,29),  // shadowmask_lv3.png
};
// todo: add support to other address mode.
float3 GetSprite(sampler2D texIn, float2 uv, uint4 rect) {
    float4 v = rect / float2(SPRITE_WIDTH, SPRITE_HEIGHT).xyxy;
    return tex2D(texIn, lerp(v.xy, v.zw, frac(uv))).rgb; // Repeat
}
// point sampler
float3 PhosphorCoatingPattern( float2 uv, uint4 rect)
{
    return GetSprite(sSpriteP, gPortrait? uv.yx:uv.xy, rect);
}
float4 PhosphorCoatingScaleLoop(out uint4 rect) // Scale && LoopUV
{
    uint2 loop[] = { gLoopSize0, gLoopSize1, gLoopSize2, gLoopSize3 };
    uint  tid[]  = {
        gPhosphor0ID[gPhosphor0],
        gPhosphor1ID[gPhosphor1],
        gPhosphor2ID[gPhosphor2],
        gPhosphor3ID[gPhosphor3] };
    int lv = gPhosphorLV;

    if(lv == 4) {
        int2 Res = OutputResolution();
        int2 lvs =
            (Res < 720)  ? (0).xx : // under HD
            (Res < 1440) ? (1).xx : // under QHD
            (Res < 2160) ? (2).xx : // under UHD(4K)
            (3).xx;  // else

        lv = gPortrait? lvs.x:lvs.y;
    }
    rect = PhosphorRects[tid[lv]];
    return float4(OutputResolution(), loop[lv]) / (rect.zw - rect.xy).xyxy;
}
// linear sampler
float3 ScanlinePattern( float2 uv )
{
    uint tid[] = {
        gScanline0ID[gScanline0],
        gScanline1ID[gScanline1],
        gScanline2ID[gScanline2]
    }; // ui indexe -> scanline id
    int lv = gScanlineLV;

    if(lv == 3) {
        int2 SrcRes = SourceResolution();
        int2 TgtRes = TargetResolution();
        int2 lvs =
            (TgtRes < SrcRes * 3) ? (0).xx : // under 3px/line
            (TgtRes < SrcRes * 6) ? (1).xx : // under 6px/line
            (2).xx; // else

        lv =  gPortrait? lvs.x : lvs.y;
    }

    return GetSprite(sSprite, gPortrait? uv.yx:uv.xy, ScanlineRects[tid[lv]]);
}

/**********************************************************
 *  helpers
 **********************************************************/

float3 lighten(float3 c0, float3 c1) { return max(c0, c1); }
float3 lighten5(float3 c0, float3 c1, float3 c2, float3 c3, float3 c4) { return max(c0,max(max(c1,c2),max(c3,c4))); }
float  brightness(float3 c) { return max(c.r, max( c.g, c.b)); }
float  luminance(float3 c) { return sqrt(dot(c, 0.3333)); }

float3 sample4(sampler2D sampIn, float2 uv, float range)
{
    float3 lv0   = tex2D(sampIn, uv).rgb;
    float4x4 lv1 = transpose(float4x4(
        tex2DgatherR(sampIn, uv) * range,
        tex2DgatherG(sampIn, uv) * range,
        tex2DgatherB(sampIn, uv) * range,
        0,0,0,0
    ));
    return lighten5(lv0, lv1[0].rgb, lv1[1].rgb, lv1[2].rgb, lv1[3].rgb);
}
float3 sample4(sampler2D sampIn, float2 tSize, float2 uv, float range)
{
    float3 offset1 = float3(tSize, 0);
    float3 lv0     = tex2D(sampIn, uv).rgb;
    float3 lv1_0   = tex2D(sampIn, uv + offset1.xz).rgb * range;
    float3 lv1_1   = tex2D(sampIn, uv + offset1.zy).rgb * range;
    float3 lv1_2   = tex2D(sampIn, uv - offset1.xz).rgb * range;
    float3 lv1_3   = tex2D(sampIn, uv - offset1.zy).rgb * range;

    return lighten5(lv0, lv1_0, lv1_1, lv1_2, lv1_3);
    //return (lv0 + lv1_0 + lv1_1 + lv1_2 + lv1_3) / 5.f;
}
float4 vs_alignTL(uint id, float2 sIn, float2 sOut, out float2 uv)
{
    uint idx = id/2;
    int2 p   = int2(idx, id - idx * 2);
    uv = p * sIn;

    p.y = -p.y;
    return float4(p * 2 * sOut - float2(1,-1), 0, 1); // top left alignment
}

// center
float4 vs_align(uint id, float2 sIn, float2 sOut, out float2 uv)
{
    uint idx = id/2;
    int2 p   = int2(idx, id - idx * 2);
    uv = p * sIn; // top left uv

    return float4( (p - .5) * float2(2,-2) * sOut, 0, 1); // center alignment
}

/**********************************************************
 *  ScreenCopy
 **********************************************************/

#define sampleMain(a) (tex2D(sLinear, (a)).rgb)

float2  CopyResolution() { return SourceResolution() / BUFFER_FSIZE; }

// input(screen) size to source size
float4 vs_copy( uint id : SV_VERTEXID, out float2 uv : TEXCOORD) : SV_POSITION
{
    return vs_alignTL(id, 1, CopyResolution(), uv);
}
float3 ps_copy( float4 pos : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET
{
    return sampleMain(uv);
}
#undef sampleMain

/**********************************************************
 *  LineScanning (Scanline simulation)
 **********************************************************/

#define sampleMain(a) (tex2D(sPoint, (a)).rgb)

// local shader inputs
bool    Crosstalk()             { return gCrosstalk; }
int     Flicker()               { return (gFramecount % 2) * 2 - 1; }
float   LuminanceCrosstalk()    { return gLumCross; }
float   ColorCrosstalk()        { return gColCross; }
float3  Gain()                  { return gGain; }
float2  ScanDir()               { return gPortrait? float2(0,1):float2(1,0); }
int2    SrcResolution()         { return gPortrait? SourceResolution().yx:SourceResolution(); }
float   GammaCorrection()       { return gGamma; }

float2  VScanNormRes()
{
    int2   source = SourceResolution();
    int2   target = TargetResolution();
    int2   screen = ScreenResolution();
    int2   iScale = float2(all(target > 0)? target : screen) / source;
    float2 hScale = round(1. + (iScale - 1.)  * (1. - gHBlur));

    // Scales texture "digitally" on the V direction, then scales "analogly" on the H direction.
    // This is a simulation of scanning lines with analog voltage signals.

    int4   res = int4( source * iScale, (int2)round( source * hScale.yx / 2) * 2 );
    return (gPortrait? res.xw : res.zy) / BUFFER_FSCAN;
}

// BT.601 Conversion Matrix
// RGB to YUV
static const float3x3 rgb2YuvMtx = float3x3(
    0.299,   0.587,   0.144,
   -0.14713,-0.28886, 0.436,
    0.615,  -0.51499,-0.10001
);
// YUV to RGB
static const float3x3 yuv2RgbMtx = float3x3(
    1.0, 0.0,     1.13983,
    1.0,-0.39465,-0.5806,
    1.0, 2.03211, 0.0
);
float3 toYUV(float3 rgb) { return mul(rgb2YuvMtx, rgb); }
float3 toRGB(float3 yuv) { return mul(yuv2RgbMtx, yuv); }
//
float3 crosstalk( float flip, float3 dYUV )
{
    return float3(LuminanceCrosstalk() * (dYUV.y + dYUV.z), ColorCrosstalk() * dYUV.xx) * flip;
}
float3 crosstalk(float3 c, float2 p)
{
    const float2 scanDir        = ScanDir();
    const float2 srcResolution  = BUFFER_FSIZE;   // sample tex res. SrcResolution();

    const int    iV             = (int)(dot(p, (1.).xx - scanDir) * srcResolution.y);
    const int    flip           = ((int)((iV % 2) * 2) -1) * Flicker();
    const float  fU             = 1./srcResolution.x;

    const float3 yuv0       = toYUV(c);
    const float3 yuvF1      = toYUV(sampleMain(p + scanDir * -1 * fU));
    const float3 yuvF2      = toYUV(sampleMain(p + scanDir * -2 * fU));
    const float3 yuvB1      = toYUV(sampleMain(p + scanDir *  1 * fU));
    const float3 yuvB2      = toYUV(sampleMain(p + scanDir *  2 * fU));

    const float3 yCbCr = yuv0 +
        crosstalk(flip, yuv0 - yuvF1) + 0.25 * crosstalk(-flip, yuvF1 - yuvF2) +
        crosstalk(flip, yuv0 - yuvB1) + 0.25 * crosstalk(-flip, yuvB1 - yuvB2);

    return toRGB(yCbCr);
}
// source size to vscan size
float4 vs_scan( uint id : SV_VERTEXID, out float2 uv : TEXCOORD) : SV_POSITION
{
    return vs_alignTL(id, CopyResolution(), VScanNormRes(), uv);
}
float3 ps_scan( float4 pos : SV_POSITION, float2 uv : TEXCOORD ) : SV_TARGET
{
    float3 c = sampleMain(uv);
    if(Crosstalk()) c = crosstalk(c, uv);

    float l = brightness(c);

    c *= saturate(ScanlinePattern(uv * SrcResolution().y) + Gain());
    return c * pow(l, 1./GammaCorrection())/l;
}
#undef sampleMain

/**********************************************************
 *  Scaling
 *      The source image is scaled to output resolution in 2 part.
 *      The reason why, an output will have less moires than it's scaled 1 part in
 *      non-integral multiples and we can easily implement over/underscan settings.
 **********************************************************/

#define sampleMain(a) (tex2D(sScan, uv).rgb)

float2  ScanScale()       { // scan scale & scan offset
    float2 invTotalScanScale;
    {
        float2 invScaleForAspectRatio = float2(OutputResolution()) / TargetResolution();
        float2 invScanScale = (bool2(gFlipX, gFlipY)? (-1.).xx:(1.).xx) / gScanScale;
        invTotalScanScale = invScanScale * invScaleForAspectRatio;
    }
    return invTotalScanScale; // auto center
}

// vscan size to output(screen) size
float4 vs_scale( uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION
{
    float4 p = vs_alignTL(id, 1, NormOutputRes(), uv);
    uv = lerp(0, VScanNormRes(), (uv - .5) * ScanScale() + .5 );
    return p;
}
// shift & scale (or use boardered tex)
float3 ps_scale( float4 pos : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET
{
    return sampleMain(uv);
}
#undef sampleMain

/**********************************************************
 *  CreateBuffer
 *      Copies the scanlined image to a work buffer.
 **********************************************************/

float   LightnessCompression()  { return 1./gIteration; }

// write to downsampled region.
float4 vs_buff( uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION
{
    return vs_alignTL(id, NormOutputRes(), NormOutputRes() * float2(.5,1), uv);
}
// Create a buffer having a margin of additive blending.
float3 ps_buff( float4 pos : SV_POSITION, float2 uv : TEXCOORD) : SV_TARGET
{
    return sample4(sLinear, uv, 0).rgb * LightnessCompression();
}

/**********************************************************
 *  Downsampling
 **********************************************************/

float Divergence() { return gDiverge; }

// naive layout
float4 vs_down( uint id, const int s, out float2 uv)
{
    if(gIteration <= s) return uv = 0, float4(0,0,-2,1);

    const float2 scale = exp2(-int2(s, s+1));
    const float2 shift = 1 - scale;
    const float2 range = NormOutputRes() * float2(1,2);

    uint idx = id/2;
    int2 p   = int2(idx, id - idx * 2);
    uv = p * scale.x * range * .5;
    uv.x += shift.x;

    p.y = -p.y;
    return float4( p * scale.y * range + float2( shift.y * 2 - 1, 1), 0, 1);
}
float3 ps_downA( float4 pos : SV_POSITION, float2 uv : TEXCOORD ) : SV_TARGET { return sample4(sBuffA, uv, Divergence()); }
float3 ps_downB( float4 pos : SV_POSITION, float2 uv : TEXCOORD ) : SV_TARGET { return sample4(sBuffB, uv, Divergence()); }
// lods
float4 vs_down0(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 0, uv); } // halfres
float4 vs_down1(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 1, uv); } // quater
float4 vs_down2(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 2, uv); } // octane
float4 vs_down3(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 3, uv); } // hexa
float4 vs_down4(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 4, uv); }
float4 vs_down5(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 5, uv); }
float4 vs_down6(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 6, uv); }
float4 vs_down7(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 7, uv); }
float4 vs_down8(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 8, uv); }
float4 vs_down9(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_down(id, 9, uv); }

/**********************************************************
 *  Upsampling
 **********************************************************/

// naive layout
float4 vs_up( uint id, const uint s, out float2 uv)
{
    if(gIteration <= s) return uv = 0, float4(0,0,-2,1);

    const float2 scale = exp2(-int2(s,s+1));
    const float2 shift = 1 - scale;
    const float2 range = NormOutputRes() * float2(1,2);

    uint idx = id/2;
    int2 p   = int2(idx, id - idx * 2);
    uv = p * scale.y * range * .5;
    uv.x += shift.y;

    p.y = -p.y;
    return float4(p * scale.x * range + float2( shift.x * 2 - 1, 1), 0, 1);
}
float3 ps_upA( float4 pos : SV_POSITION, float2 uv : TEXCOORD ) : SV_TARGET { return sample4(sBuffA, uv, 0); }
float3 ps_upB( float4 pos : SV_POSITION, float2 uv : TEXCOORD ) : SV_TARGET { return sample4(sBuffB, uv, 0); }
// lods
float4 vs_up9(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 9, uv); }
float4 vs_up8(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 8, uv); }
float4 vs_up7(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 7, uv); }
float4 vs_up6(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 6, uv); }
float4 vs_up5(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 5, uv); }
float4 vs_up4(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 4, uv); }
float4 vs_up3(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 3, uv); } // hexa
float4 vs_up2(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 2, uv); } // octane
float4 vs_up1(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 1, uv); } // quater
float4 vs_up0(uint id : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION { return vs_up(id, 0, uv); } // halfres

/**********************************************************
 *  Blending
 **********************************************************/

#define sampleMain(a) (tex2D(sBuffA, (a)).rgb)

float Intensity()  { return gIntensity; }
float Saturation() { return gSaturation; }

float4 vs_blend( uint vid : SV_VERTEXID, out float4 uv : TEXCOORD ) : SV_POSITION
{
    float4 pos = vs_align(vid, NormOutputRes(), NormOutputRes(), uv.xy); // xy sample from back buffer.
    uv.w = uv.y;
    uv.z = uv.x * .5;
    return pos;
}
// Blending to the source buffer
float3 ps_blend( float4 pos : SV_POSITION, float4 uv : TEXCOORD ) : SV_TARGET
{
    float3 divergence = saturate(sampleMain(uv.zw) * Intensity());

    float3 col = tex2D(sLinear, uv.xy).rgb;

    // luminance or brightness?
    divergence = divergence * luminance(divergence) ;
    //divergence = divergence * brightness(divergence) ;

    float4 rect, scaleLoop = PhosphorCoatingScaleLoop(rect);
    float2 picker = (uv.xy * scaleLoop.xy) % scaleLoop.zw;
    float3 mask = saturate(PhosphorCoatingPattern(picker, rect) + Saturation());

    // blend the linescanned image with its divergence using the 'Lighten(Color)' blend method.
    return lighten(col, divergence) * mask;
}
#undef sampleMain

/**********************************************************
 *  Mapping
 **********************************************************/

float4 vs_map( uint vid : SV_VERTEXID, out float2 uv : TEXCOORD ) : SV_POSITION
{
    // relative target area in output rect
    float2 target  = float2(TargetResolution()) / OutputResolution();
           target /= max(max(target.x, target.y), 1);

    // output size relative to frame buffer.
    float2 output = NormOutputRes();

    // target area relative to frame buffer
    float2 content = output * target;
    float2 map = content; // noop
    switch(gMapping)
    {
        case 1 : map = 1; break;                          // stretch
        case 2 : map /= max(map.x, map.y); break;   // contain
        case 3 : map /= min(map.x, map.y); break;   // cover
    }
    // content uv is in center of screen.
    float4 p = vs_align(vid, 1, map, uv);
    uv = (uv - .5) * content + .5;
    return p;
}
float3 ps_map( float4 pos : SV_POSITION, float2 uv : TEXCOORD ) : SV_TARGET
{
    return tex2D(sLinear, uv).rgb;
}

/**********************************************************
 *  technique
 **********************************************************/

#define QUAD PrimitiveTopology = TRIANGLESTRIP; VertexCount = 4
#define PASS_DESC(vs, ps, t) QUAD; VertexShader = vs; PixelShader = ps; RenderTarget = t

technique NaturalCRT
{
    pass donscan    { VertexShader = vs_copy;  PixelShader = ps_copy; QUAD; } // downsample source rect to source resolution.
    pass scanline   { PASS_DESC(vs_scan, ps_scan, tScan); ClearRenderTargets=true; } // tSource to VScan
    pass scaling    { VertexShader = vs_scale; PixelShader = ps_scale; QUAD; } // vscan to output size

    // setup donsample buffer (half res render region.)
    pass buff       { PASS_DESC(vs_buff, ps_buff, tBuffA); ClearRenderTargets=true; }

    // repeatdily downsample by iteration count.
    pass downsample { PASS_DESC(vs_down0, ps_downA, tBuffB); ClearRenderTargets=true; }
    pass downsample { PASS_DESC(vs_down1, ps_downB, tBuffA); }
    pass downsample { PASS_DESC(vs_down2, ps_downA, tBuffB); }
    pass downsample { PASS_DESC(vs_down3, ps_downB, tBuffA); }
    pass downsample { PASS_DESC(vs_down4, ps_downA, tBuffB); }
    pass downsample { PASS_DESC(vs_down5, ps_downB, tBuffA); }
    pass downsample { PASS_DESC(vs_down6, ps_downA, tBuffB); }
    pass downsample { PASS_DESC(vs_down7, ps_downB, tBuffA); }
    pass downsample { PASS_DESC(vs_down8, ps_downA, tBuffB); }
    pass downsample { PASS_DESC(vs_down9, ps_downB, tBuffA); }

    pass upsample   { PASS_DESC(vs_up9, ps_upA, tBuffB); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up8, ps_upB, tBuffA); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up7, ps_upA, tBuffB); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up6, ps_upB, tBuffA); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up5, ps_upA, tBuffB); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up4, ps_upB, tBuffA); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up3, ps_upA, tBuffB); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up2, ps_upB, tBuffA); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up1, ps_upA, tBuffB); BlendEnable=true; DestBlend=ONE; }
    pass upsample   { PASS_DESC(vs_up0, ps_upB, tBuffA); BlendEnable=true; DestBlend=ONE; }

    pass blend      { VertexShader = vs_blend;  PixelShader = ps_blend; QUAD; ClearRenderTargets=true;}
    // fill, contain, stretch, original
    pass mapping    { VertexShader = vs_map; PixelShader = ps_map; QUAD; ClearRenderTargets=true;}
}