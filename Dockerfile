FROM python:3.13.5-slim-bookworm AS base

# Setup env
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PATH=/home/ftuser/.local/bin:$PATH
ENV FT_APP_ENV="docker"

# Prepare environment
RUN mkdir /freqtrade \
  && apt-get update \
  && apt-get -y install sudo libatlas3-base curl sqlite3 libgomp1 \
  && apt-get clean \
  && useradd -u 1000 -G sudo -U -m -s /bin/bash ftuser \
  && chown ftuser:ftuser /freqtrade \
  # Allow sudoers
  && echo "ftuser ALL=(ALL) NOPASSWD: /bin/chown" >> /etc/sudoers

WORKDIR /freqtrade

# Install dependencies
FROM base AS python-deps
RUN  apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc \
  && apt-get clean \
  && pip install --upgrade pip wheel

# Instala TA-Lib 0.6.4 desde código fuente
COPY build_helpers/ta-lib-0.6.4-src.tar.gz /tmp/
COPY build_helpers/install_ta-lib.sh /tmp/

RUN chmod +x /tmp/install_ta-lib.sh \
 && cd /tmp && bash install_ta-lib.sh \
 && rm -rf /tmp/ta-lib*

ENV LD_LIBRARY_PATH=/usr/local/lib
RUN ldconfig


# Install dependencies
COPY --chown=ftuser:ftuser requirements.txt requirements-hyperopt.txt /freqtrade/
USER ftuser
RUN  pip install --no-cache-dir "numpy<3.0" \
  && pip install --no-cache-dir -r requirements-hyperopt.txt

# Copy dependencies to runtime-image
FROM base AS runtime-image
COPY --from=python-deps /usr/local/lib /usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib

COPY --from=python-deps --chown=ftuser:ftuser /home/ftuser/.local /home/ftuser/.local

USER ftuser
# Install and execute
COPY --chown=ftuser:ftuser . /freqtrade/

RUN pip install -e . --user --no-cache-dir --no-build-isolation \
  && mkdir /freqtrade/user_data/ \
  && freqtrade install-ui

RUN ls -la /usr/local/lib | grep ta_lib || echo "ta_lib not found, continuing..."


ENTRYPOINT ["freqtrade"]
# Default to trade mode
CMD [ "trade" ]
