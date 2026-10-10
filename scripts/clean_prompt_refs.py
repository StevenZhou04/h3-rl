"""Clean ref2va leftovers from the prompt pool: resolve @图N reference tags into their descriptions (Chinese prompt:
the prompt's own parenthetical, else a translation of refs[].desc; English reward_prompt: refs[].desc), drop the
'high frame rate, 60 fps' phrases (H3 renders 24 fps) and the 'must match the input images' clause."""
import json, re, sys
ZH = {  # refs[].desc -> Chinese, for tags whose prompt carries no parenthetical description
 "3D Tang-dynasty dancer in vermilion robe with long water sleeves": "身穿朱红长袍、甩着长水袖的3D唐代舞者",
 "3D animated Latina dancer in red ruffled dress": "身穿红色荷叶边舞裙的3D动画拉丁女舞者",
 "3D animated male dancer in white shirt and black vest": "身穿白衬衫和黑马甲的3D动画男舞者",
 "3D cartoon golden hamster in pink sweatband and sneakers": "戴粉色运动头带、穿运动鞋的3D卡通金色仓鼠",
 "3D cartoon white rabbit in red hanfu holding a lantern": "身穿红色汉服、提着灯笼的3D卡通白兔",
 "anime Indian dancer woman in turquoise sari with yellow umbrella": "身穿松石绿纱丽、撑着黄伞的动漫印度女舞者",
 "anime Mongolian girl in blue deel robe with horse-head fiddle": "身穿蓝色蒙古袍、手持马头琴的动漫蒙古族少女",
 "anime mermaid pop star with teal tail and pearl crown": "长着青绿色鱼尾、戴珍珠王冠的动漫美人鱼歌星",
 "anime pirate girl in red tricorn hat and long coat, accordion": "戴红色三角帽、穿长外套、拉着手风琴的动漫海盗少女",
 "anime punk girl with green mohawk and ripped denim vest on skateboard": "留绿色莫西干头、穿破洞牛仔马甲、踩着滑板的动漫朋克少女",
 "anime teen breakdancer in yellow hoodie and cap": "穿黄色连帽衫、戴棒球帽的动漫少年霹雳舞者",
 "anime teen in red jersey number 7 with headband": "穿红色7号球衣、戴发带的动漫少年",
 "anime teen pilot with white-blue flight suit": "身穿白蓝色飞行服的动漫少年驾驶员",
 "black cat pianist in grey fedora and suspenders, noir cartoon style": "戴灰色软呢帽、系背带的黑白卡通风格黑猫钢琴手",
 "bright pastel hamster gym with wheels and mirrors": "摆满跑轮和镜子、色调明亮柔和的仓鼠健身房",
 "bright toy-like restaurant kitchen": "明亮的玩具风格餐厅后厨",
 "candlelit medieval tavern": "烛光摇曳的中世纪酒馆",
 "candlelit ruined ballroom with broken chandelier": "烛光照亮、吊灯破碎的废弃舞厅",
 "chibi chef boy with tall white hat and red apron": "戴高高白色厨师帽、系红围裙的Q版小厨师男孩",
 "chrome humanoid DJ robot with glowing cyan visor": "戴发光青色面罩的镀铬人形DJ机器人",
 "chubby orange 3D cartoon cat in red hoodie holding drumsticks": "穿红色连帽衫、握着鼓槌的胖乎乎3D卡通橘猫",
 "county hospital corridor at night": "夜晚的县医院走廊",
 "cozy garage band room with drum kit": "摆着架子鼓的温馨车库排练室",
 "empty concrete skatepark at dusk": "黄昏时空无一人的水泥滑板公园",
 "empty subway platform at night": "深夜空荡的地铁站台",
 "flat-vector city skyline in teal and orange": "青橙配色的扁平矢量风城市天际线",
 "flat-vector girl with yellow raincoat and umbrella": "穿黄色雨衣、撑着伞的扁平矢量风女孩",
 "giant white-and-orange mecha in a hangar": "白橙相间的巨型机甲",
 "gothic anime count in crimson coat": "身穿深红大衣的哥特动漫伯爵",
 "gothic anime vampire countess in black velvet gown": "身穿黑色丝绒长裙的哥特动漫吸血鬼伯爵夫人",
 "graffiti alley with boombox": "放着大音箱的涂鸦小巷",
 "ink-wash style old tea master in grey robe": "身穿灰袍的水墨风老茶师",
 "ink-wash style swordsman in white robe with bamboo hat, rendered as brushstrokes": "以笔触呈现、身穿白袍头戴斗笠的水墨风剑客",
 "misty bamboo forest": "雾气缭绕的竹林",
 "misty mountain tea house in ink painting style": "水墨画风格的云雾山间茶舍",
 "monsoon market street with colorful awnings": "挂满彩色遮阳篷的雨季集市街道",
 "moonlit lake stage with paper swans": "漂着纸天鹅的月光湖面舞台",
 "moonlit palace terrace with lotus pond": "临着荷花池的月下宫殿露台",
 "night lantern festival street with river": "临河的夜晚灯会街道",
 "orbiting space station disco floor": "环绕轨道运行的空间站迪斯科舞池",
 "paper-cut village street with lanterns": "挂着灯笼的剪纸风村庄街道",
 "pastel storybook ballerina in white tutu": "穿白色芭蕾舞裙的柔和绘本风芭蕾舞者",
 "pixel-art fighter in blue gi with red headband": "穿蓝色道服、系红色头带的像素风格斗家",
 "pixel-art night market stage": "像素风夜市舞台",
 "red paper-cut lion head with gold edges": "金边红色剪纸狮头",
 "red-curtained teahouse stage": "挂着红色幕布的茶馆戏台",
 "rooftop street basketball court at night": "夜晚的楼顶街头篮球场",
 "smoky 1950s jazz club with grand piano": "摆着三角钢琴、烟雾缭绕的1950年代爵士俱乐部",
 "storybook bard fox with green cap and fiddle": "戴绿帽子、拉着小提琴的绘本风吟游狐狸",
 "stylized Sichuan opera performer in green-gold costume with mask": "身穿绿金戏服、戴脸谱的风格化川剧演员",
 "sunlit coral reef concert hall": "阳光洒落的珊瑚礁音乐厅",
 "sunset plaza with string lights": "挂着串灯的日落广场",
 "towering magic library": "高耸的魔法图书馆",
 "underground club with holographic crowd": "满是全息人群的地下俱乐部",
 "vast golden grassland with galloping horses": "骏马奔腾的辽阔金色草原",
 "white astronaut cat in glittery silver suit with round helmet": "穿闪亮银色宇航服、戴圆形头盔的白色宇航员猫",
 "wooden pirate ship deck at sunset": "夕阳下的木质海盗船甲板",
 'Central Asian dancer, scarlet skirt, gold bells, beaded headband': '中亚舞者，猩红长裙，金色铃铛，串珠头带',
 'Elderly tea master, long white hair, grey linen robe, calm face': '年迈的茶师，白色长发，灰色麻布长袍，神情平静',
 "Glass water tank on a scholar's desk in a study, ink drop": '书房书案上的玻璃水缸，一滴墨落入水中',
 'Heavyweight boxer, shaved head, neck tattoos, stocky build, stern face': '重量级拳击手，光头，颈部纹身，身材敦实，表情严肃',
 'K-pop girl idol, silver bob hair, black cropped jacket, neon rooftop stage': '韩流女偶像，银色波波头，黑色短款夹克，霓虹屋顶舞台',
 'Misty lake with a floating wooden platform at dawn (first frame)': '黎明时漂着木平台的雾气湖面',
 'Young male marathon runner, buzz cut, tanned skin, strong jaw, navy singlet': '年轻男子马拉松选手，寸头，晒黑的皮肤，下颌线条硬朗，藏青色背心',
 'anime schoolgirl guitarist with twin tails in navy uniform': '扎双马尾、穿藏青色校服的动漫女学生吉他手',
 'balding man in his fifties, grey Zhongshan suit, list in hand': '五十多岁的谢顶男人，灰色中山装，手里拿着名单',
 'boy of ten, tanned, patched sweater, torn sneakers': '十岁的男孩，皮肤黝黑，打着补丁的毛衣，破旧的运动鞋',
 'bright Scandinavian living room with toys on floor': '地上散落着玩具的明亮北欧风客厅',
 'clay baker bear with flour-dusted blue apron': '系着沾满面粉的蓝围裙的黏土面包师小熊',
 'gray-haired man in his sixties, mining helmet with lamp, dark work jacket': '六十多岁的白发男人，戴着带矿灯的安全帽，深色工作服',
 'guard in iron lamellar armour, red cloak, straight sabre': '身穿铁札甲、披红色斗篷、佩直刀的守卫',
 'miner in his thirties, coal-blackened face, torn overalls, bloodied leg': '三十多岁的矿工，满脸煤黑，工装破烂，腿上带血',
 'pastel clay bakery with oven and shelves': '有烤炉和货架的柔和色调黏土面包店',
 'rainy school rooftop with chain-link fence': '围着铁丝网的雨中学校天台',
 'round white robot vacuum with big cartoon eyes': '长着卡通大眼睛的圆形白色扫地机器人',
 'sixty-year-old craftsman, faded blue overalls, work cap in hand': '六十岁的老工匠，褪色的蓝色工装，手里拿着工作帽',
 'snowy wilderness battlefield': '白雪覆盖的荒野战场',
 'woman in her thirties, red scarf, padded coat, snow on shoulders': '三十多岁的女人，红围巾，棉大衣，肩上落着雪',
 'young nobleman in indigo round-collar robe, jade belt, black futou hat': '身穿靛蓝圆领袍、系玉带、戴黑色幞头的年轻贵公子',
 'young woman of twenty-three, ponytail, light-blue jacket, canvas tote': '二十三岁的年轻女子，扎马尾，浅蓝色外套，帆布包',
 'young woman with short hair, blue work uniform, red cloth in hand': '短发的年轻女子，蓝色工作服，手里拿着一块红布',
 '机械键盘，白色键帽，RGB灯效': '机械键盘，白色键帽，RGB灯效',
}

def head_zh(d): return re.split(r"[，,]", d)[0]
def head_en(d): return "the " + re.split(r",| with | in | holding | on ", d)[0]

def resolve(text, descs, head, paren, first):
    """Replace each @图N: first mention -> first(desc), later mentions -> head(desc); drop a noun the description already
    ends with when the text repeats it right after the tag (破败的@图3舞厅里 -> ...废弃舞厅里)."""
    seen = set(); out = []; pos = 0
    for m in re.finditer(paren, text):
        n = m.group(1); d = descs.get(n) or (m.group(2) if m.lastindex and m.group(2) else None)
        if d is None: raise KeyError(f"no description for @图{n}")
        rep = first(d) if n not in seen else head(d); seen.add(n)
        out.append(text[pos:m.start()]); out.append(rep); pos = m.end()
        for k in (4, 3, 2):                                   # duplicated trailing noun, e.g. 舞厅|舞厅 or ballroom|ballroom
            tail = re.split(r"[，, ]", d)[0][-k:] if re.search(r"[一-鿿]", d) else None
            if tail and len(tail) == k and text.startswith(tail, pos) and re.search(r"[一-鿿]{%d}$" % k, tail): pos += k; break
    out.append(text[pos:]); return "".join(out)

def first_zh(d):
    parts = re.split(r"[，,]", d, maxsplit=1)
    return parts[0] + (f"（{parts[1]}）" if len(parts) > 1 and parts[1].strip() else "")

def first_en(d): return "the " + d

def clean(p):
    p = dict(p); refs = {r["tag"][2:]: r["desc"] for r in p.get("refs") or []}
    t = p["prompt"]
    if "@图" in t:
        zh = {m.group(1): m.group(2) for m in re.finditer(r"@图(\d+)（([^）]*)）", t)}
        for n, d in refs.items():
            if n not in zh:
                z = ZH.get(d, d)
                if re.search(r"[A-Za-z]{3}", z) and d not in ZH: raise KeyError(f"{p['pid']}: no Chinese description for {d!r}")
                zh[n] = z
        t = resolve(t, zh, head_zh, r"@图(\d+)(?:（([^）]*)）)?", first_zh)
    t = t.replace("高帧率，60帧/秒，", "").replace("角色形象、场景设定和风格必须和输入图保持一致，", "")
    p["prompt"] = t
    e = p.get("reward_prompt")
    if e:
        e = re.sub(r"\b(the|a|an) (@图\d+)", r"\2", e, flags=re.I)
        if "@图" in e: e = resolve(e, refs, head_en, r"@图(\d+)(?:\(([^)]*)\))?", first_en)
        e = e.replace(", high frame rate, 60 fps.", ".").replace("Character design, scene setting and style must stay consistent with the input images; ", "")
        p["reward_prompt"] = e
    for k in ("prompt", "reward_prompt"):
        if p.get(k) and re.search(r"@图|60帧|60 fps|输入图|input image", p[k]): raise ValueError(f"{p['pid']} {k} still has: {p[k][:200]}")
    return p

if __name__ == "__main__":
    src = sys.argv[1]; rows = [json.loads(l) for l in open(src)]; out = [clean(r) for r in rows]
    open(src, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"{sum(a != b for a, b in zip(rows, out))} of {len(rows)} rows changed")
