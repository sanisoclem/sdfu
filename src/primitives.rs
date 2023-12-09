//! A collection of primitive SDFs that may the modified using functions in `mods`
//! or combined using functions in `ops`. Note that these primitives are always
//! centered around the origin and that you must transform the point you are sampling
//! into 'primitive-local' space. Functions are provided in `mods` to do this easier.
//!
//! Also note that while all translation and rotation transformations of the input point
//! will work properly, scaling modifies the Euclidian space and therefore does not work
//! normally. Utility function shave been provided to `Translate`, `Rotate`, and `Scale`
//! in the `SDF` trait and the `mods` module.
use crate::mathtypes::*;
use crate::SDF;
use std::marker::PhantomData;
use std::ops::*;

/// A sphere centered at origin with a radius.
#[derive(Clone, Copy, Debug)]
pub struct Sphere<T> {
    pub radius: T,
}

impl<T> Sphere<T> {
    pub fn new(radius: T) -> Self {
        Sphere { radius }
    }
}

impl<T, V> SDF<T, V> for Sphere<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec3<T>,
{
    fn dist(&self, p: V) -> T {
        p.magnitude() - self.radius
    }
}

/// A box centered at origin with axis-aligned dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Box<V, D> {
    pub dims: V,
    _pd: PhantomData<D>,
}

impl<V, D> Box<V, D> {
    pub fn new(dims: V) -> Self {
        Box {
            dims,
            _pd: PhantomData,
        }
    }
}

impl<T, V> SDF<T, V> for Box<V, Dim3D>
where
    T: Add<T, Output = T> + MaxMin + Zero + Copy,
    V: Vec3<T> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let d = p.abs() - self.dims;
        d.max(V::zero()).magnitude() + d.y().max(d.z()).max(d.x()).min(T::zero())
    }
}

impl<T, V> SDF<T, V> for Box<V, Dim2D>
where
    T: Add<T, Output = T> + MaxMin + Zero + Copy,
    V: Vec2<T> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let d = p.abs() - self.dims;
        d.max(V::zero()).magnitude() + d.y().max(d.x()).min(T::zero())
    }
}

/// A circle centered at origin with a radius.
#[derive(Clone, Copy, Debug)]
pub struct Circle<T> {
    pub radius: T,
}

impl<T> Circle<T> {
    pub fn new(radius: T) -> Self {
        Circle { radius }
    }
}

impl<T, V> SDF<T, V> for Circle<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec2<T>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        p.magnitude() - self.radius
    }
}

/// A torus that sits on the XZ plane. Thickness is the radius of
/// the wrapped cylinder while radius is the radius of the donut
/// shape.
#[derive(Clone, Copy, Debug)]
pub struct Torus<T> {
    pub radius: T,
    pub thickness: T,
}

impl<T> Torus<T> {
    pub fn new(radius: T, thickness: T) -> Self {
        Torus { radius, thickness }
    }
}

impl<T, V> SDF<T, V> for Torus<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec3<T>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let q = V::Vec2::new(
            V::Vec2::new(p.x(), p.z()).magnitude() - self.thickness,
            p.y(),
        );
        q.magnitude() - self.radius
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// An infinite cylinder extending along an axis.
#[derive(Clone, Copy, Debug)]
pub struct Cylinder<T> {
    pub radius: T,
    pub axis: Axis,
}

impl<T> Cylinder<T> {
    pub fn new(radius: T, axis: Axis) -> Self {
        Cylinder { radius, axis }
    }
}

impl<T, V> SDF<T, V> for Cylinder<T>
where
    T: Sub<T, Output = T> + Copy,
    V: Vec3<T>,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let (a, b) = match self.axis {
            Axis::X => (p.y(), p.z()),
            Axis::Y => (p.x(), p.z()),
            Axis::Z => (p.x(), p.y()),
        };
        V::Vec2::new(a, b).magnitude() - self.radius
    }
}

/// A capped cylinder extending along an axis.
#[derive(Clone, Copy, Debug)]
pub struct CappedCylinder<T> {
    pub radius: T,
    pub height: T,
    pub axis: Axis,
}

impl<T> CappedCylinder<T> {
    pub fn new(radius: T, height: T, axis: Axis) -> Self {
        CappedCylinder {
            radius,
            height,
            axis,
        }
    }
}

impl<T, V> SDF<T, V> for CappedCylinder<T>
where
    T: Sub<T, Output = T> + Add<T, Output = T> + Zero + MaxMin + Copy,
    V: Vec3<T>,
{
    #[inline]
    #[allow(clippy::many_single_char_names)]
    fn dist(&self, p: V) -> T {
        let (a, b, c) = match self.axis {
            Axis::X => (p.y(), p.z(), p.x()),
            Axis::Y => (p.x(), p.z(), p.y()),
            Axis::Z => (p.x(), p.y(), p.z()),
        };
        let d = V::Vec2::new(V::Vec2::new(a, b).magnitude(), c).abs()
            - V::Vec2::new(self.radius, self.height);
        d.x().max(d.y()).min(T::zero()) + d.max(V::Vec2::zero()).magnitude()
    }
}

/// A capsule extending from `a` to `b` with thickness `thickness`.
#[derive(Clone, Copy, Debug)]
pub struct Line<T, V> {
    pub a: V,
    pub b: V,
    pub thickness: T,
}

impl<T, V> Line<T, V> {
    pub fn new(a: V, b: V, thickness: T) -> Self {
        Line { a, b, thickness }
    }
}

impl<T, V> SDF<T, V> for Line<T, V>
where
    T: Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T> + Zero + One + Clamp + Copy,
    V: Vec<T> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> T {
        let pa = p - self.a;
        let ba = self.b - self.a;
        let t = pa.dot(ba) / ba.dot(ba);
        let h = t.clamp(T::zero(), T::one());
        (pa - (ba * h)).magnitude() - self.thickness
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle<V, D> {
    pub points: [V; 3],
    _pd: PhantomData<D>,
}

impl<V, D> Triangle<V, D> {
    pub fn new(points: [V; 3]) -> Self {
        Triangle {
            points,
            _pd: PhantomData,
        }
    }
}

impl<V> SDF<f32, V> for Triangle<V, Dim2D>
where
    V: Vec2<f32> + Copy,
{
    #[inline]
    fn dist(&self, p: V) -> f32 {
        let p0 = self.points[0];
        let p1 = self.points[1];
        let p2 = self.points[2];
        let e0 = p1 - p0;
        let e1 = p2 - p1;
        let e2 = p0 - p2;

        let v0 = p - p0;
        let v1 = p - p1;
        let v2 = p - p2;

        let pq0 = v0 - e0 * (v0.dot(e0) / e0.dot(e0)).clamp(0., 1.);
        let pq1 = v1 - e1 * (v1.dot(e1) / e1.dot(e1)).clamp(0., 1.);
        let pq2 = v2 - e2 * (v2.dot(e2) / e2.dot(e2)).clamp(0., 1.);

        let s = (e0.x() * e2.y() - e0.y() * e2.x()).signum();
        let d = <V::Vec2 as Vec2<f32>>::new(pq0.dot(pq0), s * (v0.x() * e0.y() - v0.y() * e0.x()))
            .min(<V::Vec2 as Vec2<f32>>::new(
                pq1.dot(pq1),
                s * (v1.x() * e1.y() - v1.y() * e1.x()),
            ))
            .min(<V::Vec2 as Vec2<f32>>::new(
                pq2.dot(pq2),
                s * (v2.x() * e2.y() - v2.y() * e2.x()),
            ));
        -d.x().sqrt() * d.y().signum()
    }
}
