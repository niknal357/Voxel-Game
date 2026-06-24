use bevy::prelude::*;

use super::config::TERRAIN_HEIGHT;

#[derive(Clone, Copy)]
pub(super) struct TerrainSample {
    pub(super) height: f32,
    pub(super) shade: f32,
}

pub(super) fn terrain_sample(unit: Vec3) -> TerrainSample {
    let height = terrain_height(unit);
    let slope = 1.0 - unit.dot(Vec3::Y).abs() * 0.15;
    let shade_raw = (0.78 + 0.22 * value_noise(unit * 45.0)).clamp(0.58, 1.0) * slope;
    TerrainSample {
        height,
        shade: quantize_float(shade_raw.clamp(0.55, 1.0), 10),
    }
}

fn terrain_height(unit: Vec3) -> f32 {
    // let continents = fbm(unit * 2.1 + Vec3::new(17.0, -31.0, 8.0), 5);
    // let hills = fbm(unit * 8.0 + Vec3::new(-4.0, 19.0, 52.0), 4);
    // let ridges = (1.0 - fbm(unit * 14.0 + Vec3::new(91.0, 7.0, -23.0), 4).abs()).powi(2);
    // 12.0 + continents * 34.0 + hills * 14.0 + ridges * TERRAIN_HEIGHT
    0.0
}

pub(super) fn terrain_color(tint: [u8; 3], column: TerrainSample, altitude: f32) -> [u8; 4] {
    // The GPU tree only has 254 palette entries per uploaded voxel tree. Keep
    // the procedural planet material-driven: 4 materials * 10 shade bands = at
    // most 40 colors per tile/chunk instead of one unique color per voxel.
    let material = if column.height > 72.0 || altitude > column.height - 8.0 && column.height > 55.0
    {
        0
    } else if column.height > 38.0 {
        1
    } else if column.height < -4.0 {
        2
    } else {
        3
    };

    let base = match material {
        0 => Vec3::new(218.0, 225.0, 220.0),
        1 => Vec3::new(112.0, 105.0, 88.0),
        2 => Vec3::new(45.0, 96.0, 56.0),
        _ => Vec3::new(68.0, 140.0, 72.0),
    };

    let tint = Vec3::new(tint[0] as f32, tint[1] as f32, tint[2] as f32);
    let color = (base * 0.88 + tint * 0.12) * column.shade;
    [
        quantize_u8(color.x.clamp(0.0, 255.0) as u8, 32),
        quantize_u8(color.y.clamp(0.0, 255.0) as u8, 32),
        quantize_u8(color.z.clamp(0.0, 255.0) as u8, 32),
        255,
    ]
}

fn quantize_float(value: f32, levels: u8) -> f32 {
    let max_level = (levels - 1) as f32;
    (value * max_level).round() / max_level
}

fn quantize_u8(value: u8, step: u8) -> u8 {
    ((value as u16 + (step as u16 / 2)) / step as u16 * step as u16).min(255) as u8
}

pub(super) fn tile_tint(index: usize) -> [u8; 3] {
    let hue = fract(index as f32 * 0.618_034);
    let c = hsv_to_rgb(hue, 0.25, 1.0);
    [c.x as u8, c.y as u8, c.z as u8]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
    let h = h * 6.0;
    let i = h.floor();
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Vec3::new(r * 255.0, g * 255.0, b * 255.0)
}

fn fbm(mut p: Vec3, octaves: usize) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += value_noise(p) * amp;
        norm += amp;
        p *= 2.03;
        amp *= 0.5;
    }
    sum / norm
}

fn value_noise(p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - Vec3::splat(15.0)) + Vec3::splat(10.0));

    let ix = i.x as i32;
    let iy = i.y as i32;
    let iz = i.z as i32;

    let x00 = lerp(hash3(ix, iy, iz), hash3(ix + 1, iy, iz), u.x);
    let x10 = lerp(hash3(ix, iy + 1, iz), hash3(ix + 1, iy + 1, iz), u.x);
    let x01 = lerp(hash3(ix, iy, iz + 1), hash3(ix + 1, iy, iz + 1), u.x);
    let x11 = lerp(
        hash3(ix, iy + 1, iz + 1),
        hash3(ix + 1, iy + 1, iz + 1),
        u.x,
    );
    let y0 = lerp(x00, x10, u.y);
    let y1 = lerp(x01, x11, u.y);
    lerp(y0, y1, u.z)
}

fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let mut n = x as u32;
    n = n.wrapping_mul(0x9E37_79B9) ^ (y as u32).wrapping_mul(0x85EB_CA6B);
    n ^= (z as u32).wrapping_mul(0xC2B2_AE35);
    n ^= n >> 16;
    n = n.wrapping_mul(0x7FEB_352D);
    n ^= n >> 15;
    n = n.wrapping_mul(0x846C_A68B);
    n ^= n >> 16;
    (n as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}
