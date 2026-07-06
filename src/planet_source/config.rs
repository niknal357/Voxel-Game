use voxel_streaming::CHUNK_SIZE;

pub(super) const PLANET_TILE_COUNT: usize = 1024;
pub(super) const RANDOM_PLANET_TILE_ORIENTATION: bool = true;
pub(crate) const PLANET_RADIUS: f32 = 1024.0 * 32.0;
pub(super) const PLANET_COST: u32 = 1;

// Each tangent grid is clipped by the spherical Voronoi cell around its
// Fibonacci point. These are only radial limits; x/y bounds are inferred from
// the cached Voronoi halfspaces per tile.
pub(super) const TILE_INWARD_DEPTH: i32 = 3000;
pub(super) const TILE_OUTWARD_HEIGHT: i32 = 512;
pub(super) const TILE_BOUND_PADDING: i32 = CHUNK_SIZE * 2;
pub(super) const VORONOI_NEIGHBORS: usize = 32;
pub(super) const TERRAIN_HEIGHT: f32 = 200.0;
pub(super) const TILE_SHAPE_EPSILON: f32 = 0.001;
