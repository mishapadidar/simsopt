#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>


#include "cachedarray.h"
#include "biot_savart_impl.h"
#include "curve.h"
#include "current.h"
#include "coil.h"

using std::logic_error;
using std::vector;
using std::shared_ptr;
using std::make_shared;

template<class Array>
class MagneticField {
    /*
     * This is the abstract base class for a magnetic field B and it's potential A.
     * The usage is as follows:
     * Bfield = InstanceOfMagneticField(...)
     * points = some array of shape (n, 3) where to evaluate the B field
     * Bfield.set_points(points)
     * B = Bfield.B() // to get the magnetic field at `points`, a (n, 3) array
     * A = Bfield.A() // to get the potential field at `points`, a (n, 3) array
     * gradB = Bfield.dB_by_dX() // to get the gradient of the magnetic field at `points`, a (n, 3, 3) array
     * Some performance notes:
     *  - this class has an internal cache that is cleared everytime set_points() is called
     *  - all functions have a `_ref` version, e.g. `Bfield.B_ref()` which
     *    returns a reference to the array in the cache. this should be used when
     *    performance is key and when the user guarantees that the array is only
     *    read and not modified.
     */
    private:
        std::map<string, CachedArray<Array>> cache;

    public:
        Array points;

        MagneticField() {
            points = xt::zeros<double>({1, 3});
        }

        bool cache_get_status(string key){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found
                return false;
            }
            if(!(loc->second.status)){ // needs recomputing
                return false;
            }
            return true;
        }

        Array& cache_get_or_create(string key, vector<int> dims){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            loc->second.status = true;
            return loc->second.data;
        }

        Array& cache_get_or_create_and_fill(string key, vector<int> dims, std::function<void(Array&)> impl){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            if(!(loc->second.status)){ // needs recomputing
                impl(loc->second.data);
                loc->second.status = true;
            }
            return loc->second.data;
        }

        void invalidate_cache() {
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                it->second.status = false;
            }
        }

        MagneticField& set_points(Array& p) {
            this->invalidate_cache();
            points = p;
            return *this;
        }

        virtual void B_impl(Array& B) { throw logic_error("B_impl was not implemented"); }
        virtual void dB_by_dX_impl(Array& dB_by_dX) { throw logic_error("dB_by_dX_impl was not implemented"); }
        virtual void d2B_by_dXdX_impl(Array& d2B_by_dXdX) { throw logic_error("d2B_by_dXdX_impl was not implemented"); }
        virtual void A_impl(Array& A) { throw logic_error("A_impl was not implemented"); }
        virtual void dA_by_dX_impl(Array& dA_by_dX) { throw logic_error("dA_by_dX_impl was not implemented"); }
        virtual void d2A_by_dXdX_impl(Array& d2A_by_dXdX) { throw logic_error("d2A_by_dXdX_impl was not implemented"); }

        Array& B_ref() {
            return cache_get_or_create_and_fill("B", {static_cast<int>(points.shape(0)), 3}, [this](Array& B) { return B_impl(B);});
        }
        Array& dB_by_dX_ref() {
            return cache_get_or_create_and_fill("dB_by_dX", {static_cast<int>(points.shape(0)), 3, 3}, [this](Array& dB_by_dX) { return dB_by_dX_impl(dB_by_dX);});
        }
        Array& d2B_by_dXdX_ref() {
            return cache_get_or_create_and_fill("d2B_by_dXdX", {static_cast<int>(points.shape(0)), 3, 3, 3}, [this](Array& d2B_by_dXdX) { return d2B_by_dXdX_impl(d2B_by_dXdX);});
        }
        Array B() { return B_ref(); }
        Array dB_by_dX() { return dB_by_dX_ref(); }
        Array d2B_by_dXdX() { return d2B_by_dXdX_ref(); }

        Array& A_ref() {
            return cache_get_or_create_and_fill("A", {static_cast<int>(points.shape(0)), 3}, [this](Array& A) { return A_impl(A);});
        }
        Array& dA_by_dX_ref() {
            return cache_get_or_create_and_fill("dA_by_dX", {static_cast<int>(points.shape(0)), 3, 3}, [this](Array& dA_by_dX) { return dA_by_dX_impl(dA_by_dX);});
        }
        Array& d2A_by_dXdX_ref() {
            return cache_get_or_create_and_fill("d2A_by_dXdX", {static_cast<int>(points.shape(0)), 3, 3, 3}, [this](Array& d2A_by_dXdX) { return d2A_by_dXdX_impl(d2A_by_dXdX);});
        }
        Array A() { return A_ref(); }
        Array dA_by_dX() { return dA_by_dX_ref(); }
        Array d2A_by_dXdX() { return d2A_by_dXdX_ref(); }
};

typedef vector_type AlignedVector;

template<class Array>
class BiotSavart : public MagneticField<Array> {
    /*
     * This class describes a Magnetic field induced by a list of coils. It
     * computes the Biot Savart law to evaluate the field.
     */
    private:

        vector<shared_ptr<Coil<Array>>> coils;
        // this vectors are aligned in memory for fast simd usage.
        AlignedVector pointsx = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsy = AlignedVector(xsimd::simd_type<double>::size, 0.);
        AlignedVector pointsz = AlignedVector(xsimd::simd_type<double>::size, 0.);

        void fill_points(const Array& points) {
            int npoints = points.shape(0);
            // allocating these aligned vectors is not super cheap, so reuse
            // whenever possible.
            if(pointsx.size() != npoints)
                pointsx = AlignedVector(npoints, 0.);
            if(pointsy.size() != npoints)
                pointsy = AlignedVector(npoints, 0.);
            if(pointsz.size() != npoints)
                pointsz = AlignedVector(npoints, 0.);
            for (int i = 0; i < npoints; ++i) {
                pointsx[i] = points(i, 0);
                pointsy[i] = points(i, 1);
                pointsz[i] = points(i, 2);
            }
        }

    public:
        using MagneticField<Array>::points;
        using MagneticField<Array>::cache_get_or_create;
        using MagneticField<Array>::cache_get_status;
        BiotSavart(vector<shared_ptr<Coil<Array>>> coils) : coils(coils) {

        }

        void compute(int derivatives) {
            fmt::print("Calling compute({})\n", derivatives);
            this->fill_points(points);
            Array dummyjac = xt::zeros<double>({1, 1, 1});
            Array dummyhess = xt::zeros<double>({1, 1, 1, 1});
            int npoints = static_cast<int>(points.shape(0));
            int ncoils = this->coils.size();
            Array& B = cache_get_or_create("B", {npoints, 3});
            Array& dB = derivatives > 0 ? cache_get_or_create("dB_by_dX", {npoints, 3, 3}) : dummyjac;
            Array& ddB = derivatives > 1 ? cache_get_or_create("d2B_by_dXdX", {npoints, 3, 3, 3}) : dummyhess;

            B *= 0; // TODO Actually set to zero, multiplying with zero doesn't get rid of NANs
            dB *= 0;
            ddB *= 0;

            // annoyingly computing gamma and gammadash in JaxHelicalCurve
            // doesn't seem safe to do in parallel, so we do these calls here
            // once to make sure the cache is filled.
            for (int i = 0; i < ncoils; ++i) {
                this->coils[i]->curve->gamma();
                this->coils[i]->curve->gammadash();
            }
            //fmt::print("B(0, :) = ({}, {}, {}) at {}\n", B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
#pragma omp parallel for
            for (int i = 0; i < ncoils; ++i) {
                Array& Bi = cache_get_or_create(fmt::format("B_{}", i), {npoints, 3});
                Bi *= 0;
                Array& gamma = this->coils[i]->curve->gamma();
                Array& gammadash = this->coils[i]->curve->gammadash();
                double current = this->coils[i]->current->get_value();
                if(derivatives == 0){
                    biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dummyjac, dummyhess);
                } else {
                    Array& dBi = cache_get_or_create(fmt::format("dB_{}", i), {npoints, 3, 3});
                    dBi *= 0;
                    if(derivatives == 1) {
                        biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, dummyhess);
                    } else {
                        Array& ddBi = cache_get_or_create(fmt::format("ddB_{}", i), {npoints, 3, 3, 3});
                        ddBi *= 0;
                        if (derivatives == 2) {
                            biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gamma, gammadash, Bi, dBi, ddBi);
                        } else {
                            throw logic_error("Only two derivatives of Biot Savart implemented");
                        }
                        ////fmt::print("ddBi(0, 0, 0, :) = ({}, {}, {})\n", ddBi(0, 0, 0, 0), ddBi(0, 0, 0, 1), ddBi(0, 0, 0, 2));
#pragma omp critical
                        {
                            xt::noalias(ddB) = ddB + current * ddBi;
                        }
                    }
#pragma omp critical
                    {
                        xt::noalias(dB) = dB + current * dBi;
                    }
                }
                //fmt::print("i={}, Bi(0, :) = ({}, {}, {}) at {}\n", i, Bi(0, 0), Bi(0, 1), Bi(0, 2), fmt::ptr(B.data()));
                //fmt::print("i={},  B(0, :) = ({}, {}, {}) at {}\n", i, B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
#pragma omp critical
                {
                    xt::noalias(B) = B + current * Bi;
                }
                //fmt::print("i={},  B(0, :) = ({}, {}, {}) at {}\n", i, B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
            }
            //fmt::print("B(0, :) = ({}, {}, {}) at {}\n", B(0, 0), B(0, 1), B(0, 2), fmt::ptr(B.data()));
        }


        void B_impl(Array& B) override {
            this->compute(0);
        }
        
        void dB_by_dX_impl(Array& dB_by_dX) override {
            this->compute(1);
        }

        void d2B_by_dXdX_impl(Array& d2B_by_dXdX) override {
            this->compute(2);
        }

        //Array dB_by_dcoeff_vjp(Array& vec) {
        //    int num_coils = this->coilcollection.curves.size();
        //    Array dummy = Array();
        //    auto res_gamma = std::vector<Array>(num_coils, Array());
        //    auto res_gammadash = std::vector<Array>(num_coils, Array());
        //    auto res_current = std::vector<Array>(num_coils, Array());
        //    for(int i=0; i<num_coils; i++) {
        //        int num_points = this->coils[i].curve.gamma().shape(0);
        //        res_gamma[i] = xt::zeros<double>({num_points, 3});
        //        res_gammadash[i] = xt::zeros<double>({num_points, 3});
        //        res_current[i] = xt::zeros<double>({1});
        //    }
        //    this->fill_points(points);
        //    for(int i=0; i<num_coils; i++) {
        //            biot_savart_vjp_kernel<Array, 0>(pointsx, pointsy, pointsz, this->coilcollection.curves[i].gamma(), this->coilcollection.curves[i].gammadash(),
        //                    vec, res_gamma[i], res_gammadash[i], dummy, dummy, dummy);
        //    }

        //    int npoints = points.shape(0);
        //    for(int i=0; i<num_coils; i++) {
        //        Array& Bi = cache_get_or_create(fmt::format("B_{}", i), {static_cast<int>(points.shape(0)), 3});
        //        for (int j = 0; j < npoints; ++j) {
        //            res_current[i] += Bi(j, 0)*vec(j, 0) + Bi(j, 1)*vec(j, 1) + Bi(j, 2)*vec(j, 2);
        //        }
        //    }

        //    // TODO: figure out how to add these in the right way, when some correspond to coil dofs, others correspond to coil currents etc
        //    return this->coilcollection.dgamma_by_dcoeff_vjp(res_gamma)
        //        + this->coilcollection.dgammadash_by_dcoeff_vjp(res_gammadash)
        //        + this->coilcollection.dcurrent_by_dcoeff_vjp(res_current);

        //}

};


