#ifndef JULIA_COMPLEX_H
#define JULIA_COMPLEX_H

// TODO: Сделать шаблонным классом.

/**
 * Класс представляющий комплексное число.
 * Копирующий конструктор не требуется.
 */
class Complex {
    float r = 0.;
    float i = 0.;

public:
    /**
     * Конструктор.
     *
     * @param r Реальная часть комплексного числа.
     * @param i Мнимая часть комплексного числа.
     */
    __device__ Complex(float r, float i): r(r), i(i) {}

    /**
     * Конструктор создающий комплексное значение из числа с плавающей запятой.
     * Делегирующий конструктор. Должен вызываться явно.
     *
     * @param r Реальная часть комплексного числа.
     */
    explicit  __device__ Complex(float r): Complex{r, 0.} {}

    /**
     * Конструктор по умолчанию. Делегирующий конструктор.
     */
    __device__ Complex(): Complex{0., 0.} {}

    /**
     * Квадрат модуля комплексного числа.
     */
    __device__ float magnitude2() const {
        return r*r + i*i;
    }

    /**
     * Оператор присваивания.
     * Должен возвращать ссылку на объект, чтобы использовать цепочки присваивания.
     */
    __device__ Complex& operator =(const Complex& src) = default;

    /**
     * Оператор присваивания числа с плавающей запятой.
     * Должен возвращать ссылку на объект, чтобы использовать цепочки присваивания.
     */
    __device__ Complex& operator =(float nr) {
        r = nr; i = 0.;
        return *this;
    }

    /**
     * Умножение 2х комплексных чисел.
     */
    __device__ Complex operator *(const Complex& src) const {
        return {r*src.r - i*src.i, i*src.r + r*src.i};
    }

    /**
     * Деление 2х комплексных чисел (число делит текущее).
     */
    __device__ Complex operator /(const Complex& src) const {
        auto m2 = src.r*src.r + src.i*src.i;
        return {(r*src.r + i*src.i)/m2, (i*src.r - r*src.i)/m2};
    }

    /**
     * Сложение 2х комплексных чисел.
     */
    __device__ Complex operator +(const Complex& src) const {
        return {r + src.r, i + src.i};
    }

    /**
     * Вычитание 2х комплексных чисел (число вычитается из текущего).
     */
    __device__ Complex operator -(const Complex& src) const {
        return {r - src.r, i - src.i};
    }
};
#endif //JULIA_COMPLEX_H
