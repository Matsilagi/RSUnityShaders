#include "ReShade.fxh"

uniform float3 Pencil_Color <
	ui_type = "color";
	ui_label = "Pencil Color [Paper2]";
> = float3(0.0,0.371,0.78);

uniform float Pencil_Size <
	ui_type = "drag";
	ui_min = 0.001;
	ui_max = 0.022;
	ui_step = 0.001;
	ui_label = "Pencil Size [Paper2]";
	ui_tooltip = "The size of the Pencil";
> = 0.008;

uniform float Pencil_Correction <
	ui_type = "drag";
	ui_min = 0.0;
	ui_max = 2.0;
	ui_step = 0.001;
	ui_label = "Pencil Correction [Paper2]";
	ui_tooltip = "How much are the lines corrected.";
> = 0.76;

uniform float Intensity <
	ui_type = "drag";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_step = 0.001;
	ui_label = "Pencil Intensity [Paper2]";
	ui_tooltip = "How intense are the lines.";
> = 1.0;

uniform float Speed_Animation <
	ui_type = "drag";
	ui_min = 0.0;
	ui_max = 2.0;
	ui_step = 0.001;
	ui_label = "Speed Animation [Paper2]";
	ui_tooltip = "How fast does the lines anim.";
> = 1.0;

uniform float Corner_Lose <
	ui_type = "drag";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_step = 0.001;
	ui_label = "Corner Loss [Paper2]";
	ui_tooltip = "How much of the corners are lost.";
> = 0.85;

uniform float Fade_Paper_to_BackColor <
	ui_type = "drag";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_step = 0.001;
	ui_label = "Paper Fade [Paper2]";
	ui_tooltip = "How much does the paper blend with the OG image.";
> = 0.0;

uniform float Fade_With_Original <	
	ui_type = "drag";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_step = 0.001;
	ui_label = "Lines Fade [Paper2]";
	ui_tooltip = "How much does the lines blend with the paper.";
> = 1.0;

uniform float3 Back_Color <
	ui_type = "color";
	ui_label = "Paper Color [Paper2]";
> = float3(1.0,1.0,1.0);

//SHADER START
texture tPaper2 <source="CameraFilterPack/CameraFilterPack_Paper3.jpg";> { Width=1024; Height=1024; Format = RGBA8;};
sampler2D sPaper2 { Texture=tPaper2; MinFilter=LINEAR; MagFilter=LINEAR; MipFilter=LINEAR; AddressU = REPEAT; AddressV = REPEAT; };

uniform int _TimerX < source = "frametime"; >;

float4 PS_Paper2(float4 pos : SV_Position, float2 texcoord : TEXCOORD0) : SV_Target
{
	float4 f = tex2D(ReShade::BackBuffer, texcoord);
	float3 paper = tex2D(sPaper2,texcoord).rgb;
	float _TimeX;
	_TimeX += _TimerX;
	if (_TimeX > 100) _TimeX = 0;
	float ce = 1;
	float4 tex1[4];
	float4 tex2[4]; 
	float tex=Pencil_Size;
	float tt=_TimeX*Speed_Animation;
	float s=floor(sin(tt*10)*0.02)/12;
	float c=floor(cos(tt*10)*0.02)/12;
	float pg=1-paper.g;
	float2 dist=float2(c+paper.b*0.02,s+paper.b*0.02);
	float3 paper2 = tex2D(sPaper2,texcoord+dist).rgb;
	tex2[0] = tex2D(ReShade::BackBuffer, texcoord+float2(tex,0)+dist/128);
	tex2[1] = tex2D(ReShade::BackBuffer, texcoord+float2(-tex,0)+dist/128);
	tex2[2] = tex2D(ReShade::BackBuffer, texcoord+float2(0,tex)+dist/128);
	tex2[3] = tex2D(ReShade::BackBuffer, texcoord+float2(0,-tex)+dist/128);

	for(int i = 0; i < 4; i++) 
	{
		tex1[i] = saturate(1.0-distance(tex2[i].r, f.r));
		tex1[i] *= saturate(1.0-distance(tex2[i].g, f.g));
		tex1[i] *= saturate(1.0-distance(tex2[i].b, f.b));
		tex1[i] = pow(tex1[i], Pencil_Correction*25);
		ce *= dot(tex1[i], 1.0);
	}

	ce=saturate(ce);
	float l = 1-ce;
	float3 ax = l; 
	ax*=paper2.b;
	ax=lerp(float3(0.0,0.0,0.0),ax*Intensity*1.5,1);
	float gg=lerp(1-paper.g,0,1-Corner_Lose);
	ax=lerp(ax,float3(0.0,0.0,0.0),gg);
	paper=lerp(float3(paper.r,paper.r,paper.r),Back_Color.rgb,Fade_Paper_to_BackColor); 
	paper=lerp(paper,Pencil_Color.rgb,ax*Intensity);
	paper-=pg*0.5;
	paper = lerp(f,paper,Fade_With_Original);
	return float4(paper, 1.0);
}

//techniques///////////////////////////////////////////////////////////////////////////////////////////

technique Paper2 {
	pass Paper2 {
		VertexShader=PostProcessVS;
		PixelShader=PS_Paper2;
	}
}